"""Strategy template.

Two hooks, borrowed from the best of each framework:

* ``prepare(data)`` – vectorised indicator computation over the whole history
  once (freqtrade ``populate_indicators`` / backtesting.py ``self.I``). Only
  use rolling / past-looking operations here; the engine guarantees that
  ``on_bar`` only ever *reads* rows up to the current date.
* ``on_bar(ctx)`` – event-driven decision making on each bar
  (vn.py / backtrader / zipline style), placing orders through ``ctx``.

``params`` is a plain dict with defaults in ``default_params`` so the
optimizer (grid search / walk-forward) can sweep them, like freqtrade hyperopt.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

import pandas as pd

from ..core.context import Context

STRATEGY_REGISTRY: dict[str, type["Strategy"]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        STRATEGY_REGISTRY[name] = cls
        return cls
    return deco


def get_strategy(name: str) -> type["Strategy"]:
    from . import builtin  # noqa: F401  (populates the registry)
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"未知策略 {name!r}，可选: {', '.join(sorted(STRATEGY_REGISTRY))}")
    return STRATEGY_REGISTRY[name]


class Strategy:
    name: ClassVar[str] = "base"
    title: ClassVar[str] = ""                    # 中文名称
    default_params: ClassVar[dict[str, Any]] = {}
    param_grid: ClassVar[dict[str, list]] = {}   # suggested optimisation ranges
    warmup: ClassVar[int] = 0                    # bars needed before signals are valid

    def __init__(self, **params):
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name} 不支持参数: {sorted(unknown)}")
        self.params = {**self.default_params, **params}
        self.ind: dict[str, pd.DataFrame] = {}

    def __getattr__(self, item):
        params = self.__dict__.get("params", {})
        if item in params:
            return params[item]
        raise AttributeError(item)

    # ---------------------------------------------------------------- hooks
    def prepare(self, data: dict[str, pd.DataFrame]) -> None:
        """Pre-compute indicators into ``self.ind[symbol]``."""

    def on_start(self, ctx: Context) -> None:
        """Called once before the first bar."""

    def on_bar(self, ctx: Context) -> None:
        raise NotImplementedError

    def on_end(self, ctx: Context) -> None:
        """Called after the last bar."""

    # ---------------------------------------------------------------- helpers
    def value(self, ctx: Context, symbol: str, column: str, offset: int = 0) -> Optional[float]:
        """Indicator value at the current bar (offset=1 → previous bar)."""
        df = self.ind.get(symbol)
        if df is None:
            return None
        n = int(df.index.searchsorted(ctx.dt, side="right")) - offset
        if n <= 0 or n > len(df):
            return None
        v = df[column].iloc[n - 1]
        return None if pd.isna(v) else float(v)

    def cross_section(self, ctx: Context, column: str) -> pd.Series:
        """One indicator across all symbols that trade today (for ranking)."""
        out = {}
        for sym in self.ind:
            if not ctx.is_trading(sym):
                continue
            v = self.value(ctx, sym, column)
            if v is not None:
                out[sym] = v
        return pd.Series(out, dtype=float)

    def describe(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
