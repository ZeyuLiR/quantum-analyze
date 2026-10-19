"""Performance metrics (in the spirit of pyfolio / quantstats / empyrical)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_daily(equity: pd.Series) -> pd.Series:
    """Collapse intraday equity to one value per trading day."""
    equity = equity.dropna()
    if equity.empty:
        return equity
    return equity.groupby(equity.index.normalize()).last()


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch (in bars) spent below a previous peak."""
    under = (equity < equity.cummax()).astype(int).to_numpy()
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def compute_metrics(equity: pd.Series, trades: Optional[pd.DataFrame] = None,
                    benchmark: Optional[pd.Series] = None, risk_free: float = 0.02,
                    periods: int = TRADING_DAYS) -> dict:
    eq = to_daily(equity)
    if len(eq) < 2:
        return {}
    rets = eq.pct_change().dropna()
    years = max(len(rets) / periods, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = rets.std(ddof=1) * math.sqrt(periods)
    rf_d = (1 + risk_free) ** (1 / periods) - 1
    excess = rets - rf_d
    sharpe = excess.mean() / rets.std(ddof=1) * math.sqrt(periods) if rets.std(ddof=1) > 0 else 0.0
    downside = rets[rets < 0].std(ddof=1)
    sortino = excess.mean() / downside * math.sqrt(periods) if downside and downside > 0 else 0.0
    dd = drawdown(eq)
    mdd = dd.min()
    m = {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "final_value": float(eq.iloc[-1]),
        "total_return": float(total), "annual_return": float(cagr),
        "annual_volatility": float(vol), "sharpe": float(sharpe), "sortino": float(sortino),
        "max_drawdown": float(mdd), "max_drawdown_days": max_drawdown_duration(eq),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "daily_win_rate": float((rets > 0).mean()),
        "best_day": float(rets.max()), "worst_day": float(rets.min()),
    }
    monthly = eq.resample("ME").last().pct_change().dropna()
    if len(monthly):
        m["monthly_win_rate"] = float((monthly > 0).mean())

    if trades is not None and len(trades):
        sells = trades[trades["direction"] == "sell"]
        m["trades"] = int(len(trades))
        m["round_trips"] = int(len(sells))
        if len(sells):
            wins, losses = sells[sells["pnl"] > 0]["pnl"], sells[sells["pnl"] <= 0]["pnl"]
            m["win_rate"] = float(len(wins) / len(sells))
            m["profit_factor"] = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else math.inf
            m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
            m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        m["total_fees"] = float((trades["commission"] + trades["tax"]).sum())
        traded = (trades["price"] * trades["quantity"] * trades.get("fx", 1.0)).sum()
        m["annual_turnover"] = float(traded / eq.mean() / years / 2)

    if benchmark is not None and len(benchmark.dropna()) > 1:
        b = to_daily(benchmark).reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            br = b.pct_change().dropna()
            r, br = rets.align(br, join="inner")
            m["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
            byears = max(len(br) / periods, 1e-9)
            m["benchmark_annual_return"] = float((b.iloc[-1] / b.iloc[0]) ** (1 / byears) - 1)
            m["excess_annual_return"] = m["annual_return"] - m["benchmark_annual_return"]
            if br.var() > 0:
                beta = float(np.cov(r, br)[0, 1] / br.var())
                alpha = float(((r - rf_d) - beta * (br - rf_d)).mean() * periods)
                te = (r - br).std(ddof=1) * math.sqrt(periods)
                m.update(beta=beta, alpha=alpha,
                         information_ratio=float((r - br).mean() * periods / te) if te > 0 else 0.0)
    return m


LABELS = {
    "start": "开始日期", "end": "结束日期", "final_value": "期末资产",
    "total_return": "累计收益", "annual_return": "年化收益", "annual_volatility": "年化波动",
    "sharpe": "夏普比率", "sortino": "索提诺比率", "max_drawdown": "最大回撤",
    "max_drawdown_days": "最长回撤期(天)", "calmar": "卡玛比率", "daily_win_rate": "日胜率",
    "monthly_win_rate": "月胜率", "best_day": "最好单日", "worst_day": "最差单日",
    "trades": "成交笔数", "round_trips": "平仓次数", "win_rate": "交易胜率",
    "profit_factor": "盈亏比(总)", "avg_win": "平均盈利", "avg_loss": "平均亏损",
    "total_fees": "总交易费用", "annual_turnover": "年换手(倍)",
    "benchmark_return": "基准累计收益", "benchmark_annual_return": "基准年化",
    "excess_annual_return": "年化超额", "alpha": "Alpha", "beta": "Beta",
    "information_ratio": "信息比率",
}
PCT = {"total_return", "annual_return", "annual_volatility", "max_drawdown", "daily_win_rate",
       "monthly_win_rate", "best_day", "worst_day", "win_rate", "benchmark_return",
       "benchmark_annual_return", "excess_annual_return", "alpha"}


def format_metrics(m: dict) -> list[tuple[str, str]]:
    rows = []
    for k, v in m.items():
        label = LABELS.get(k, k)
        if isinstance(v, float):
            if k in PCT:
                s = f"{v:.2%}"
            elif abs(v) >= 1000:
                s = f"{v:,.0f}"
            else:
                s = f"{v:.2f}"
        else:
            s = str(v)
        rows.append((label, s))
    return rows
