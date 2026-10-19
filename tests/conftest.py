import numpy as np
import pandas as pd
import pytest


def make_bars(seed: int = 0, n: int = 600, start: str = "2020-01-01", drift: float = 0.0004,
              vol: float = 0.018, price: float = 20.0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    ret = r.normal(drift, vol, n) + 0.001 * np.sin(np.arange(n) / 40)
    c = price * np.exp(np.cumsum(ret))
    o = c * (1 + r.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.005, n)))
    lo = np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.005, n)))
    v = r.integers(1_000_000, 10_000_000, n).astype(float)
    df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=idx)
    df["pre_close"] = df["close"].shift(1)
    return df


@pytest.fixture
def panel():
    return {f"{600000 + i:06d}.SH": make_bars(i) for i in range(6)}
