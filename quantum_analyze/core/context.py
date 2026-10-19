"""Strategy context: the one API a strategy talks to.

Mirrors the zipline / rqalpha / JoinQuant style (``order_target_percent``,
``history``) so strategies read naturally, and it is backed by an *account*
object that is either the simulated broker (backtest, paper) or a live
gateway account. Strategy code does not change between the three modes.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Protocol

import pandas as pd

from .objects import Bar, Direction, Order, Position
from .rules import MarketRegistry

if TYPE_CHECKING:
    from ..risk import RiskManager

log = logging.getLogger("quantum_analyze")


class Account(Protocol):
    """What the context needs from an execution venue (SimBroker or LiveAccount)."""
    cash: float
    positions: dict[str, Position]
    registry: MarketRegistry

    @property
    def total_value(self) -> float: ...
    def position(self, symbol: str) -> Position: ...
    def pending_for(self, symbol: str) -> list[Order]: ...
    def submit(self, order: Order) -> Order: ...
    def reject(self, order: Order, reason: str) -> Order: ...
    def cancel_all(self, symbol: Optional[str] = None) -> None: ...


class Context:
    def __init__(self, data: dict[str, pd.DataFrame], account: Account,
                 risk: Optional["RiskManager"] = None, calendar: Optional[pd.DatetimeIndex] = None):
        self.data = data
        self.account = account
        self.risk = risk
        self.calendar = calendar if calendar is not None else _union_index(data)
        self.dt: Optional[pd.Timestamp] = None
        self.state: dict = {}          # free-form, persisted by the paper/live runner
        self.logs: list[str] = []
        self._cut: dict[str, int] = {}
        self._period_starts: dict[str, set] = {}
        self._arrays = {sym: (df.index, df[["open", "high", "low", "close", "volume", "pre_close"]]
                              .to_numpy(dtype=float))
                        for sym, df in data.items()} if data else {}

    # ------------------------------------------------------------------ clock
    def set_dt(self, dt: pd.Timestamp) -> None:
        self.dt = dt
        self._cut = {}

    def _n(self, symbol: str) -> int:
        n = self._cut.get(symbol)
        if n is None:
            df = self.data.get(symbol)
            n = 0 if df is None else int(df.index.searchsorted(self.dt, side="right"))
            self._cut[symbol] = n
        return n

    # ------------------------------------------------------------------ data access (never beyond dt)
    @property
    def universe(self) -> list[str]:
        return list(self.data)

    def history(self, symbol: str, field: str = "close", n: Optional[int] = None) -> pd.Series:
        df = self.data.get(symbol)
        if df is None:
            return pd.Series(dtype=float)
        end = self._n(symbol)
        start = 0 if n is None else max(end - n, 0)
        return df[field].iloc[start:end]

    def frame(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        df = self.data.get(symbol)
        if df is None:
            return pd.DataFrame()
        end = self._n(symbol)
        return df.iloc[0 if n is None else max(end - n, 0):end]

    def bar(self, symbol: str) -> Optional[Bar]:
        """Today's bar, or None if the symbol did not trade today (suspended)."""
        arr = self._arrays.get(symbol)
        n = self._n(symbol)
        if arr is None or n == 0 or arr[0][n - 1] != self.dt:
            return None
        o, h, l, c, v, pc = arr[1][n - 1]
        return Bar(symbol, self.dt, o, h, l, c, v, pc)

    def price(self, symbol: str) -> float:
        """Latest known close (last traded price if suspended today)."""
        arr = self._arrays.get(symbol)
        n = self._n(symbol)
        if arr is None or n == 0:
            return math.nan
        return float(arr[1][n - 1, 3])

    def is_trading(self, symbol: str) -> bool:
        b = self.bar(symbol)
        return b is not None and b.volume > 0

    def is_period_start(self, freq: str = "M") -> bool:
        """True on the first trading day of a week (W), month (M) or quarter (Q).

        With ``fill_at = next_open`` the orders made today execute tomorrow,
        i.e. still early in the new period.
        """
        freq = freq.upper()
        if freq in ("D", "DAILY"):
            return True
        if freq not in self._period_starts:
            cal = pd.DatetimeIndex(self.calendar).normalize().unique()
            key = {"W": cal.to_period("W"), "M": cal.to_period("M"), "Q": cal.to_period("Q"),
                   "Y": cal.to_period("Y")}[freq[0]]
            first = pd.Series(cal, index=key).groupby(level=0).min()
            self._period_starts[freq] = set(first.values)
        return pd.Timestamp(self.dt).normalize().to_datetime64() in self._period_starts[freq]

    # ------------------------------------------------------------------ portfolio
    @property
    def cash(self) -> float:
        return self.account.cash

    @property
    def total_value(self) -> float:
        return self.account.total_value

    @property
    def positions(self) -> dict[str, Position]:
        return self.account.positions

    def position(self, symbol: str) -> Position:
        return self.account.position(symbol)

    def weight(self, symbol: str) -> float:
        p = self.account.positions.get(symbol)
        tv = self.total_value
        if not p or tv <= 0:
            return 0.0
        return p.quantity * self.price(symbol) * self.account.registry.fx(symbol) / tv

    def log(self, msg: str) -> None:
        line = f"[{'' if self.dt is None else self.dt.date()}] {msg}"
        self.logs.append(line)
        log.info(line)

    # ------------------------------------------------------------------ ordering
    def order(self, symbol: str, quantity: int, price: Optional[float] = None,
              tag: str = "") -> Optional[Order]:
        """Buy (quantity > 0) or sell (quantity < 0). Buys are rounded down to lots."""
        if quantity == 0:
            return None
        direction = Direction.BUY if quantity > 0 else Direction.SELL
        qty = abs(int(quantity))
        rules = self.account.registry.spec(symbol).rules
        if direction == Direction.BUY:
            qty = rules.round_lot(symbol, qty)
            if qty <= 0:
                return None
        order = Order(symbol=symbol, direction=direction, quantity=qty, price=price,
                      created=self.dt, tag=tag)
        if self.risk is not None:
            ok, reason = self.risk.check_order(order, self)
            if not ok:
                self.log(f"风控拒单 {symbol} {direction.value} {qty}: {reason}")
                return self.account.reject(order, reason)
        return self.account.submit(order)

    def order_target_shares(self, symbol: str, target: int, price: Optional[float] = None,
                            tag: str = "") -> Optional[Order]:
        self.account.cancel_all(symbol)
        current = self.account.position(symbol).quantity
        delta = int(target) - current
        if delta == 0:
            return None
        if delta < 0 and target > 0:
            # partial sell: keep lots tidy (odd lots may only be sold all at once)
            rules = self.account.registry.spec(symbol).rules
            delta = -rules.round_lot(symbol, -delta)
            if delta == 0:
                return None
        return self.order(symbol, delta, price, tag)

    def order_target_value(self, symbol: str, value: float, price: Optional[float] = None,
                           tag: str = "") -> Optional[Order]:
        px = self.price(symbol)
        if not px or math.isnan(px) or px <= 0:
            return None
        fx = self.account.registry.fx(symbol)
        rules = self.account.registry.spec(symbol).rules
        target = rules.round_lot(symbol, max(value, 0.0) / (px * fx))
        if value <= 0:
            target = 0
        current = self.account.position(symbol).quantity
        # avoid churning: skip trades smaller than one lot
        if target > current and target - current < rules.lot_for(symbol):
            return None
        return self.order_target_shares(symbol, target, price, tag)

    def order_target_percent(self, symbol: str, pct: float, price: Optional[float] = None,
                             tag: str = "") -> Optional[Order]:
        return self.order_target_value(symbol, self.total_value * pct, price, tag)

    def rebalance(self, weights: dict[str, float], tag: str = "rebalance",
                  min_change: float = 0.0, cash_buffer: float = 0.01) -> None:
        """Move the portfolio to ``weights`` (fractions of total value).

        Symbols currently held but absent from ``weights`` are sold. Trades
        that change a weight by less than ``min_change`` are skipped to save
        costs. ``cash_buffer`` keeps a bit of cash for fees and slippage.
        """
        scale = 1.0 - cash_buffer
        for sym in list(self.account.positions):
            if sym not in weights:
                self.order_target_shares(sym, 0, tag=tag)
        for sym, w in sorted(weights.items(), key=lambda kv: kv[1]):
            if abs(w * scale - self.weight(sym)) < min_change and w > 0:
                continue
            self.order_target_percent(sym, w * scale, tag=tag)

    def close_all(self, tag: str = "close_all") -> None:
        for sym in list(self.account.positions):
            self.order_target_shares(sym, 0, tag=tag)


def row_to_bar(symbol: str, dt: pd.Timestamp, row: pd.Series) -> Bar:
    return Bar(symbol=symbol, datetime=dt, open=float(row["open"]), high=float(row["high"]),
               low=float(row["low"]), close=float(row["close"]),
               volume=float(row.get("volume", 0.0)),
               pre_close=float(row.get("pre_close", math.nan)))


def _union_index(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex([])
    for df in data.values():
        idx = idx.union(df.index)
    return idx
