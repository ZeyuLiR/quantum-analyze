"""Risk management: pre-trade checks and portfolio-level guards.

Inspired by vn.py's RiskManager app (pre-trade order checks) and freqtrade's
stoploss / protections (stop-loss, trailing stop, drawdown circuit breaker).
All checks run identically in backtest, paper and live modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .core.objects import Direction, Order

if TYPE_CHECKING:
    from .core.context import Context


@dataclass
class RiskConfig:
    max_position_weight: float = 1.0      # cap per symbol, fraction of total value
    max_order_value: float = 0.0          # cap per order in base currency; 0 = off
    max_daily_orders: int = 0             # 0 = off
    blacklist: list[str] = field(default_factory=list)
    stop_loss: float = 0.0                # e.g. 0.08: sell when price < cost * (1-8%)
    trailing_stop: float = 0.0            # e.g. 0.12: sell when price < peak * (1-12%)
    take_profit: float = 0.0              # e.g. 0.30: sell when price > cost * (1+30%)
    max_holding_days: int = 0             # time stop; 0 = off
    max_drawdown_halt: float = 0.0        # e.g. 0.20: stop opening positions beyond 20% DD
    halt_liquidate: bool = False          # also liquidate when the halt triggers
    halt_cooldown_days: int = 20          # trading days before buying is allowed again

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RiskConfig":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg = config or RiskConfig()
        self.peak_value = 0.0
        self.halted_until: Optional[int] = None
        self._day_index = 0
        self._orders_today = 0
        self._last_day: Optional[pd.Timestamp] = None
        self.events: list[str] = []

    # ------------------------------------------------------------------ pre-trade
    def check_order(self, order: Order, ctx: "Context") -> tuple[bool, str]:
        cfg = self.cfg
        if order.direction == Direction.SELL:
            return True, ""   # reducing risk is always allowed
        if order.symbol in cfg.blacklist:
            return False, "黑名单"
        if self.halted_until is not None and self._day_index < self.halted_until:
            return False, "回撤熔断中，禁止开仓"
        if cfg.max_daily_orders and self._orders_today >= cfg.max_daily_orders:
            return False, "超过单日委托上限"
        px = ctx.price(order.symbol) * ctx.account.registry.fx(order.symbol)
        value = order.quantity * px
        if cfg.max_order_value and value > cfg.max_order_value:
            return False, f"单笔金额 {value:,.0f} 超过上限 {cfg.max_order_value:,.0f}"
        if cfg.max_position_weight < 1.0 and ctx.total_value > 0:
            held = ctx.position(order.symbol).quantity * px
            new_w = (held + value) / ctx.total_value
            if new_w > cfg.max_position_weight + 1e-6:
                # shrink rather than reject: buy up to the cap
                rules = ctx.account.registry.spec(order.symbol).rules
                room = cfg.max_position_weight * ctx.total_value - held
                qty = rules.round_lot(order.symbol, room / px) if px > 0 else 0
                if qty <= 0:
                    return False, f"单票仓位超过 {cfg.max_position_weight:.0%}"
                order.quantity = qty
        self._orders_today += 1
        return True, ""

    # ------------------------------------------------------------------ after close
    def on_day(self, ctx: "Context", new_day: bool = True) -> None:
        """Called once per bar after prices update and before the strategy runs."""
        if new_day:
            self._day_index += 1
            self._orders_today = 0
        cfg = self.cfg
        tv = ctx.total_value
        self.peak_value = max(self.peak_value, tv)
        if cfg.max_drawdown_halt and self.peak_value > 0:
            dd = 1 - tv / self.peak_value
            if dd >= cfg.max_drawdown_halt and (self.halted_until is None or self._day_index >= self.halted_until):
                self.halted_until = self._day_index + cfg.halt_cooldown_days
                self.peak_value = tv   # re-arm from the new base after cooldown
                msg = f"组合回撤 {dd:.1%} 触发熔断，{cfg.halt_cooldown_days} 个交易日内禁止开仓"
                self.events.append(f"{ctx.dt.date()} {msg}")
                ctx.log(msg)
                if cfg.halt_liquidate:
                    ctx.close_all(tag="risk_halt")
        for sym, pos in list(ctx.positions.items()):
            reason = self._exit_reason(ctx, sym, pos)
            if reason:
                ctx.log(f"{sym} 触发{reason}，卖出")
                self.events.append(f"{ctx.dt.date()} {sym} {reason}")
                ctx.order_target_shares(sym, 0, tag=reason)

    def _exit_reason(self, ctx: "Context", sym: str, pos) -> str:
        cfg = self.cfg
        px = ctx.price(sym)
        if not px or pos.avg_cost <= 0:
            return ""
        if cfg.stop_loss and px <= pos.avg_cost * (1 - cfg.stop_loss):
            return "止损"
        if cfg.trailing_stop and pos.peak_price and px <= pos.peak_price * (1 - cfg.trailing_stop):
            return "移动止损"
        if cfg.take_profit and px >= pos.avg_cost * (1 + cfg.take_profit):
            return "止盈"
        if cfg.max_holding_days and pos.entry_date is not None:
            held = ((ctx.calendar > pos.entry_date) & (ctx.calendar <= ctx.dt)).sum()
            if held >= cfg.max_holding_days:
                return "持有到期"
        return ""

    @property
    def halted(self) -> bool:
        return self.halted_until is not None and self._day_index < self.halted_until

    def to_dict(self) -> dict:
        return {"peak_value": self.peak_value, "halted_until": self.halted_until,
                "day_index": self._day_index, "events": self.events[-200:]}

    def load(self, d: dict) -> None:
        self.peak_value = d.get("peak_value", 0.0)
        self.halted_until = d.get("halted_until")
        self._day_index = d.get("day_index", 0)
        self.events = d.get("events", [])


def position_size_by_atr(total_value: float, risk_per_trade: float, atr: float,
                         atr_multiple: float, price: float) -> float:
    """Turtle-style sizing: lose at most ``risk_per_trade`` of equity if the
    stop at ``atr_multiple`` ATRs is hit. Returns a target weight (0..1)."""
    if atr <= 0 or price <= 0 or total_value <= 0:
        return 0.0
    shares = total_value * risk_per_trade / (atr * atr_multiple)
    return min(shares * price / total_value, 1.0)
