"""Basic trading objects (orders, trades, positions).

The data model follows vn.py's object layer: plain dataclasses that every
module (broker, risk, gateway, analysis) shares, so a live gateway can be
swapped in without touching strategy code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"          # partially filled, remainder cancelled (day order)
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    direction: Direction
    quantity: int
    price: Optional[float] = None    # None = market order at the fill price
    order_id: str = ""
    created: Optional[pd.Timestamp] = None
    status: OrderStatus = OrderStatus.SUBMITTED
    filled_qty: int = 0
    avg_price: float = 0.0
    reason: str = ""                 # rejection / cancel reason, or strategy tag
    tag: str = ""

    @property
    def is_active(self) -> bool:
        return self.status == OrderStatus.SUBMITTED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["status"] = self.status.value
        d["created"] = None if self.created is None else str(self.created)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Order":
        d = dict(d)
        d["direction"] = Direction(d["direction"])
        d["status"] = OrderStatus(d["status"])
        d["created"] = None if d.get("created") is None else pd.Timestamp(d["created"])
        return cls(**d)


@dataclass
class Trade:
    trade_id: str
    order_id: str
    symbol: str
    direction: Direction
    quantity: int
    price: float
    commission: float
    tax: float
    datetime: pd.Timestamp
    pnl: float = 0.0     # realized pnl net of fees (sells only)
    tag: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["datetime"] = None if self.datetime is None else str(self.datetime)
        return d


@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0        # includes buy-side fees
    today_bought: int = 0        # A-share T+1: shares bought today are not sellable
    last_price: float = 0.0
    entry_date: Optional[pd.Timestamp] = None
    peak_price: float = 0.0      # highest close since entry, for trailing stops

    @property
    def sellable(self) -> int:
        return max(self.quantity - self.today_bought, 0)

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_cost) * self.quantity

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_date"] = None if self.entry_date is None else str(self.entry_date)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        d["entry_date"] = None if d.get("entry_date") is None else pd.Timestamp(d["entry_date"])
        return cls(**d)


@dataclass
class Bar:
    symbol: str
    datetime: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    pre_close: float = float("nan")
    extra: dict = field(default_factory=dict)
