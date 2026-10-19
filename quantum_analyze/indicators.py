"""Vectorised technical indicators and price-volume factors.

All functions only look backwards (rolling / ewm / shift(+n)), so they are
safe to pre-compute over the whole history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": 2 * (dif - dea)})


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0).where(up.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                      (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    return pd.DataFrame({"mid": mid, "upper": mid + k * std, "lower": mid - k * std,
                         "pct_b": (close - (mid - k * std)) / (2 * k * std)})


def donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Channel of the *previous* n bars (excludes today, so a breakout is today's close above it)."""
    return pd.DataFrame({"upper": df["high"].rolling(n, min_periods=n).max().shift(1),
                         "lower": df["low"].rolling(n, min_periods=n).min().shift(1)})


def momentum(close: pd.Series, n: int, skip: int = 0) -> pd.Series:
    """n-bar return, optionally skipping the most recent ``skip`` bars."""
    return close.shift(skip) / close.shift(n) - 1


def volatility(close: pd.Series, n: int = 20, annualize: int = 252) -> pd.Series:
    return close.pct_change().rolling(n, min_periods=n).std() * np.sqrt(annualize)


def trend_r2_slope(close: pd.Series, n: int = 25) -> pd.Series:
    """Annualised log-price regression slope times R² (a robust momentum score)."""
    y = np.log(close)
    x = np.arange(n, dtype=float)
    xm = x - x.mean()
    denom = (xm ** 2).sum()

    def _score(win: np.ndarray) -> float:
        if np.isnan(win).any():
            return np.nan
        ym = win - win.mean()
        slope = (xm * ym).sum() / denom
        ss_tot = (ym ** 2).sum()
        r2 = 0.0 if ss_tot == 0 else (slope * xm).dot(slope * xm) / ss_tot
        return (np.exp(slope * 252) - 1) * r2

    return y.rolling(n, min_periods=n).apply(_score, raw=True)


def zscore_cross(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score with 3-sigma winsorisation."""
    s = s.astype(float)
    if s.std(ddof=0) == 0 or len(s) < 2:
        return s * 0
    z = (s - s.mean()) / s.std(ddof=0)
    return z.clip(-3, 3)
