"""Data source interface.

Every source returns, per symbol, a DataFrame indexed by a tz-naive
``DatetimeIndex`` (ascending, unique) with float columns::

    open, high, low, close, volume [, amount] [, pre_close]

* prices are *forward adjusted* (前复权) unless the source is configured
  otherwise; ``volume`` is in shares (not 手); ``amount`` in the market's
  currency.
* symbols use the ``CODE.MARKET`` convention (``600519.SH``, ``00700.HK``,
  ``AAPL.US``); each source converts to its own code format internally.
* ``frequency`` is ``"1d"`` (daily) or a minute bar such as ``"1m"``, ``"5m"``,
  ``"15m"``, ``"30m"``, ``"60m"``.

Sources that need an optional third-party package import it lazily inside
``fetch`` and raise ``DataSourceUnavailable`` with an install hint, so the
core package only depends on numpy and pandas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ..core.rules import market_of, normalize_symbol

COLUMNS = ["open", "high", "low", "close", "volume"]


class DataSourceError(Exception):
    """Fetching failed (network error, bad symbol, empty result...)."""


class DataSourceUnavailable(DataSourceError):
    """The source cannot be used here (package missing, no token, no client running)."""


class DataSource(ABC):
    name: str = "base"
    markets: tuple[str, ...] = ("CN", "HK", "US")   # markets this source can serve
    frequencies: tuple[str, ...] = ("1d",)

    def supports(self, symbol: str, frequency: str = "1d") -> bool:
        return market_of(symbol) in self.markets and frequency in self.frequencies

    @abstractmethod
    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        """Return normalised bars for one symbol (see module docstring)."""

    def load(self, symbols: list[str], start: Optional[str] = None, end: Optional[str] = None,
             frequency: str = "1d") -> dict[str, pd.DataFrame]:
        out = {}
        for s in symbols:
            sym = normalize_symbol(s)
            out[sym] = normalize_bars(self.fetch(sym, start, end, frequency))
        return out


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a bar frame: sort, dedupe, float columns, pre_close."""
    if df is None or df.empty:
        raise DataSourceError("空数据")
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(f"缺少列 {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    keep = COLUMNS + [c for c in ("amount", "pre_close") if c in df.columns]
    df = df[keep].astype(float)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if "pre_close" not in df.columns:
        df["pre_close"] = df["close"].shift(1)
    else:
        df["pre_close"] = df["pre_close"].fillna(df["close"].shift(1))
    df.index.name = "datetime"
    return df


def clip_dates(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return df
