"""Parameter optimisation: grid search and walk-forward analysis.

Grid search alone over-fits; walk-forward (train on a rolling window, pick
the best parameters, then trade them on the *following* unseen window) is the
honest estimate of live performance. Prefer parameters whose neighbours also
score well (a plateau) over a single sharp peak.
"""
from __future__ import annotations

import itertools
import logging
from typing import Optional

import pandas as pd

from .engine import run_backtest
from .risk import RiskConfig
from .strategies.base import Strategy

log = logging.getLogger("quantum_analyze")


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _valid(cls: type[Strategy], params: dict) -> bool:
    # common sanity constraint: fast < slow style pairs
    for a, b in (("fast", "slow"), ("exit_n", "entry_n")):
        if a in params and b in params and params[a] >= params[b]:
            return False
    return True


def grid_search(cls: type[Strategy], data: dict[str, pd.DataFrame], grid: dict[str, list],
                start: Optional[str] = None, end: Optional[str] = None,
                objective: str = "sharpe", base_params: Optional[dict] = None,
                min_trades: int = 0, **bt_kwargs) -> pd.DataFrame:
    rows = []
    for params in expand_grid(grid):
        full = {**(base_params or {}), **params}
        if not _valid(cls, full):
            continue
        res = run_backtest(cls(**full), data, start=start, end=end, **bt_kwargs)
        m = res.metrics
        if m.get("trades", 0) < min_trades:
            continue
        rows.append({**params, **{k: m.get(k) for k in (
            "annual_return", "sharpe", "max_drawdown", "calmar", "sortino", "win_rate", "trades")}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(objective, ascending=False, na_position="last").reset_index(drop=True)


def walk_forward(cls: type[Strategy], data: dict[str, pd.DataFrame], grid: dict[str, list],
                 train_months: int = 36, test_months: int = 12, objective: str = "sharpe",
                 start: Optional[str] = None, end: Optional[str] = None,
                 base_params: Optional[dict] = None, initial_cash: float = 1_000_000.0,
                 risk: Optional[RiskConfig] = None, **bt_kwargs) -> dict:
    """Rolling walk-forward; returns windows table, stitched out-of-sample NAV and metrics."""
    from .analysis.metrics import compute_metrics

    idx = _union(data)
    first = pd.Timestamp(start) if start else idx[0]
    last = pd.Timestamp(end) if end else idx[-1]
    windows, nav_parts = [], []
    nav_level = 1.0
    t0 = first
    while True:
        train_end = t0 + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), last)
        if test_start > last:
            break
        table = grid_search(cls, data, grid, start=str(t0.date()), end=str(train_end.date()),
                            objective=objective, base_params=base_params,
                            initial_cash=initial_cash, risk=risk, **bt_kwargs)
        if table.empty:
            break
        best = {k: _py(table.at[0, k]) for k in grid}
        res = run_backtest(cls(**{**(base_params or {}), **best}), data, start=str(test_start.date()),
                           end=str(test_end.date()), initial_cash=initial_cash, risk=risk, **bt_kwargs)
        eq = res.equity["total"]
        nav = eq / eq.iloc[0] * nav_level
        nav_level = float(nav.iloc[-1])
        nav_parts.append(nav)
        windows.append({"train": f"{t0.date()}~{train_end.date()}",
                        "test": f"{test_start.date()}~{test_end.date()}",
                        **best, f"train_{objective}": table.iloc[0][objective],
                        "test_return": res.metrics.get("total_return"),
                        f"test_{objective}": res.metrics.get(objective),
                        "test_max_drawdown": res.metrics.get("max_drawdown")})
        log.info("walk-forward %s best=%s", windows[-1]["test"], best)
        t0 = t0 + pd.DateOffset(months=test_months)
        if test_end >= last:
            break
    oos = pd.concat(nav_parts) if nav_parts else pd.Series(dtype=float)
    oos = oos[~oos.index.duplicated(keep="first")]
    return {"windows": pd.DataFrame(windows), "oos_nav": oos,
            "oos_metrics": compute_metrics(oos) if len(oos) > 1 else {}}


def _union(data):
    idx = pd.DatetimeIndex([])
    for df in data.values():
        idx = idx.union(df.index)
    return idx


def _py(v):
    return v.item() if hasattr(v, "item") else v
