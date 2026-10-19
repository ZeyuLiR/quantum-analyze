"""Simulated broker: order matching with A-share rules.

Used both for backtesting and paper trading (its state round-trips through
``to_dict``/``from_dict``). Orders are day orders: whatever cannot fill on
the matching bar is cancelled with a reason, like a real exchange at close.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable, Optional

import pandas as pd

from .objects import Bar, Direction, Order, OrderStatus, Position, Trade
from .rules import MarketRegistry


class SimBroker:
    """Cash is kept in the registry's base currency; prices stay in each
    market's own currency and are converted with ``registry.fx``."""

    def __init__(self, initial_cash: float = 1_000_000.0,
                 registry: Optional[MarketRegistry] = None):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.registry = registry or MarketRegistry()
        self.positions: dict[str, Position] = {}
        self.pending: list[Order] = []
        self.orders: list[Order] = []
        self.trades: list[Trade] = []
        self._order_seq = itertools.count(1)
        self._trade_seq = itertools.count(1)
        self.current_dt: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------ state
    @property
    def market_value(self) -> float:
        return sum(p.market_value * self.registry.fx(s) for s, p in self.positions.items())

    @property
    def total_value(self) -> float:
        return self.cash + self.market_value

    def position(self, symbol: str) -> Position:
        return self.positions.get(symbol) or Position(symbol)

    def pending_for(self, symbol: str) -> list[Order]:
        return [o for o in self.pending if o.symbol == symbol]

    # ------------------------------------------------------------------ orders
    def submit(self, order: Order) -> Order:
        order.order_id = order.order_id or f"O{next(self._order_seq):07d}"
        order.created = order.created or self.current_dt
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = order.reason or "数量必须为正"
        self.orders.append(order)
        if order.status == OrderStatus.SUBMITTED:
            self.pending.append(order)
        return order

    def reject(self, order: Order, reason: str) -> Order:
        order.order_id = order.order_id or f"O{next(self._order_seq):07d}"
        order.created = order.created or self.current_dt
        order.status = OrderStatus.REJECTED
        order.reason = reason
        self.orders.append(order)
        return order

    def cancel(self, order: Order, reason: str = "撤单") -> None:
        if order.is_active:
            order.status = OrderStatus.CANCELLED
            order.reason = reason
        if order in self.pending:
            self.pending.remove(order)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        for o in list(self.pending):
            if symbol is None or o.symbol == symbol:
                self.cancel(o)

    # ------------------------------------------------------------------ day cycle
    def on_day_start(self, dt: pd.Timestamp) -> None:
        """Call on every new trading day (not every intraday bar): unlocks T+1 shares."""
        self.current_dt = dt
        for p in self.positions.values():
            p.today_bought = 0

    def set_time(self, dt: pd.Timestamp) -> None:
        self.current_dt = dt

    def update_prices(self, bars: dict[str, Bar]) -> None:
        for sym, p in self.positions.items():
            bar = bars.get(sym)
            if bar is not None and not math.isnan(bar.close):
                p.last_price = bar.close
                p.peak_price = max(p.peak_price, bar.close)

    def match(self, bars: dict[str, Bar], price_field: str = "open") -> list[Trade]:
        """Match all pending orders against ``bars``; sells first to free cash."""
        fills: list[Trade] = []
        orders = sorted(self.pending, key=lambda o: 0 if o.direction == Direction.SELL else 1)
        for order in orders:
            trade = self._match_one(order, bars.get(order.symbol), price_field)
            if trade is not None:
                fills.append(trade)
        # day orders: anything left is cancelled
        for o in list(self.pending):
            self.cancel(o, o.reason or "当日未成交")
        self.pending = []
        return fills

    def _match_one(self, order: Order, bar: Optional[Bar], price_field: str) -> Optional[Trade]:
        if bar is None or bar.volume <= 0 or math.isnan(getattr(bar, price_field)):
            order.reason = "停牌/无行情"
            return None
        ref = getattr(bar, price_field)
        is_buy = order.direction == Direction.BUY
        spec = self.registry.spec(order.symbol)
        rules, cost = spec.rules, spec.cost
        fx = self.registry.fx(order.symbol)
        lo, hi = rules.limit_prices(order.symbol, bar.pre_close)
        if is_buy and ref >= hi - 1e-9:
            order.reason = "涨停无法买入"
            return None
        if not is_buy and ref <= lo + 1e-9:
            order.reason = "跌停无法卖出"
            return None

        price = ref * (1 + cost.slippage) if is_buy else ref * (1 - cost.slippage)
        price = min(max(price, bar.low, lo), bar.high, hi)
        price = round(round(price / rules.tick) * rules.tick, 6)
        if order.price is not None:
            if is_buy and price > order.price + 1e-9:
                order.reason = "限价未触及"
                return None
            if not is_buy and price < order.price - 1e-9:
                order.reason = "限价未触及"
                return None

        qty = order.quantity
        if rules.max_volume_ratio > 0 and bar.volume > 0:
            cap = bar.volume * rules.max_volume_ratio
            if qty > cap:
                qty = int(cap) if not is_buy else rules.round_lot(order.symbol, cap)

        pos = self.positions.get(order.symbol)
        if is_buy:
            qty = rules.round_lot(order.symbol, qty)
            step = max(rules.lot_for(order.symbol), 1)
            while qty > 0:
                commission, tax = cost.fees(True, qty * price, qty)
                if (qty * price + commission + tax) * fx <= self.cash + 1e-6:
                    break
                qty -= step
            if qty <= 0:
                order.reason = "资金不足"
                return None
        else:
            available = pos.sellable if pos else 0
            qty = min(qty, available)
            if qty <= 0:
                order.reason = "无可卖持仓(T+1)" if pos and pos.quantity else "无持仓"
                return None

        commission, tax = cost.fees(is_buy, qty * price, qty)
        trade = Trade(
            trade_id=f"T{next(self._trade_seq):07d}", order_id=order.order_id,
            symbol=order.symbol, direction=order.direction, quantity=qty, price=price,
            commission=commission, tax=tax, datetime=self.current_dt, tag=order.tag,
        )
        self._apply(trade, fx, rules.t_plus)
        order.filled_qty = qty
        order.avg_price = price
        order.status = OrderStatus.FILLED if qty == order.quantity else OrderStatus.PARTIAL
        if order.status == OrderStatus.PARTIAL:
            order.reason = "部分成交，余量撤销"
        self.pending.remove(order)
        self.trades.append(trade)
        return trade

    def _apply(self, t: Trade, fx: float = 1.0, t_plus: int = 1) -> None:
        pos = self.positions.setdefault(t.symbol, Position(t.symbol))
        fees = t.commission + t.tax
        if t.direction == Direction.BUY:
            new_qty = pos.quantity + t.quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + t.price * t.quantity + fees) / new_qty
            if pos.quantity == 0:
                pos.entry_date = t.datetime
                pos.peak_price = t.price
            pos.quantity = new_qty
            if t_plus >= 1:
                pos.today_bought += t.quantity
            self.cash -= (t.price * t.quantity + fees) * fx
        else:
            t.pnl = (t.price - pos.avg_cost) * t.quantity - fees
            pos.quantity -= t.quantity
            self.cash += (t.price * t.quantity - fees) * fx
        pos.last_price = t.price if pos.last_price == 0 else pos.last_price
        if pos.quantity == 0:
            del self.positions[t.symbol]

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "pending": [o.to_dict() for o in self.pending],
            "trades": [t.to_dict() for t in self.trades],
            "order_seq": next(self._order_seq),
            "trade_seq": next(self._trade_seq),
            "current_dt": None if self.current_dt is None else str(self.current_dt),
        }

    @classmethod
    def from_dict(cls, d: dict, registry: Optional[MarketRegistry] = None) -> "SimBroker":
        b = cls(d["initial_cash"], registry)
        b.cash = d["cash"]
        b.positions = {s: Position.from_dict(p) for s, p in d["positions"].items()}
        b.pending = [Order.from_dict(o) for o in d["pending"]]
        b.orders = list(b.pending)
        for t in d.get("trades", []):
            t = dict(t)
            t["direction"] = Direction(t["direction"])
            t["datetime"] = None if t.get("datetime") in (None, "None") else pd.Timestamp(t["datetime"])
            b.trades.append(Trade(**t))
        b._order_seq = itertools.count(d.get("order_seq", 1))
        b._trade_seq = itertools.count(d.get("trade_seq", 1))
        b.current_dt = None if d.get("current_dt") is None else pd.Timestamp(d["current_dt"])
        return b

    def holdings(self) -> Iterable[Position]:
        return self.positions.values()
