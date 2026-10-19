"""Broker gateway interface (live / broker-side paper accounts).

A gateway connects to a real broker API. ``LiveAccount`` adapts any gateway
to the same ``Account`` protocol the simulated broker implements, so a
strategy runs unchanged in backtest → paper → live (vn.py's gateway idea).

Gateways only need these primitives; symbols use the ``CODE.MARKET``
convention and each gateway converts to its broker's own codes.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..core.objects import Direction, Order, OrderStatus, Position
from ..core.rules import MarketRegistry

log = logging.getLogger("quantum_analyze.gateway")


class GatewayError(Exception):
    pass


class GatewayUnavailable(GatewayError):
    """Required SDK not installed or client (TWS / OpenD / QMT) not running."""


@dataclass
class AccountSnapshot:
    cash: float              # available cash in base currency
    total_value: float       # net liquidation value in base currency
    currency: str = "CNY"


class BaseGateway(ABC):
    name: str = "base"
    markets: tuple[str, ...] = ("CN", "HK", "US")
    paper: bool = False      # True when connected to the broker's simulated account

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def query_account(self) -> AccountSnapshot: ...

    @abstractmethod
    def query_positions(self) -> dict[str, Position]:
        """Positions keyed by ``CODE.MARKET``; ``today_bought`` must reflect
        unsellable (T+1) shares where the broker reports them."""

    @abstractmethod
    def send_order(self, order: Order) -> str:
        """Send an order and return the broker order id.

        ``order.price is None`` means "marketable": gateways should use a
        market order where allowed, else a limit order at a protective price.
        """

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> None: ...

    def query_open_orders(self) -> list[dict]:
        return []

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()


class LiveAccount:
    """Account protocol on top of a gateway. Orders are *sent immediately*.

    With ``dry_run=True`` orders are only recorded (printed for review), which
    is the safe default for a first run against a real account.
    """

    def __init__(self, gateway: BaseGateway, registry: Optional[MarketRegistry] = None,
                 dry_run: bool = True, limit_offset: float = 0.0):
        self.gateway = gateway
        self.registry = registry or MarketRegistry()
        self.dry_run = dry_run
        # >0: turn market orders into protective limit orders at ref*(1±offset)
        self.limit_offset = limit_offset
        self.price_ref = None   # callable(symbol) -> reference price, set by the runner
        self.orders: list[Order] = []
        self.pending: list[Order] = []
        self.refresh()

    def refresh(self) -> None:
        snap = self.gateway.query_account()
        self.cash = snap.cash
        self._total = snap.total_value
        self.positions: dict[str, Position] = self.gateway.query_positions()
        self.pending = []   # broker is the source of truth after a refresh

    @property
    def total_value(self) -> float:
        return self._total

    def position(self, symbol: str) -> Position:
        return self.positions.get(symbol) or Position(symbol)

    def pending_for(self, symbol: str) -> list[Order]:
        return [o for o in self.pending if o.symbol == symbol]

    def submit(self, order: Order) -> Order:
        order.order_id = order.order_id or f"L{len(self.orders) + 1:06d}"
        if order.direction == Direction.SELL:
            avail = self.position(order.symbol).sellable
            if avail <= 0:
                return self.reject(order, "无可卖持仓(T+1)")
            order.quantity = min(order.quantity, avail)
        if order.price is None and self.limit_offset > 0 and self.price_ref is not None:
            ref = self.price_ref(order.symbol)
            if ref and ref > 0:
                tick = self.registry.spec(order.symbol).rules.tick
                k = 1 + self.limit_offset if order.direction == Direction.BUY else 1 - self.limit_offset
                order.price = round(round(ref * k / tick) * tick, 6)
        self.orders.append(order)
        if self.dry_run:
            log.warning("[DRY-RUN] %s %s %d @ %s", order.direction.value, order.symbol,
                        order.quantity, order.price or "MKT")
            order.reason = "dry-run 未发送"
            self.pending.append(order)
            return order
        try:
            order.order_id = self.gateway.send_order(order)
            self.pending.append(order)
        except Exception as e:  # keep running other orders, record failure
            order.status = OrderStatus.REJECTED
            order.reason = f"发送失败: {e}"
            log.error("send_order failed: %s", e)
        return order

    def reject(self, order: Order, reason: str) -> Order:
        order.status = OrderStatus.REJECTED
        order.reason = reason
        self.orders.append(order)
        return order

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        for o in list(self.pending):
            if symbol is None or o.symbol == symbol:
                if not self.dry_run and o.order_id:
                    try:
                        self.gateway.cancel_order(o.order_id)
                    except Exception as e:
                        log.error("cancel failed: %s", e)
                o.status = OrderStatus.CANCELLED
                self.pending.remove(o)
