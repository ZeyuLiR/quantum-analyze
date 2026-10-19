"""Event-driven bar engine shared by backtest, paper and live trading.

Bar loop (one iteration per bar, daily or intraday)::

    new trading day?  -> unlock T+1 shares
    fill_at=next_open -> match yesterday's orders at this bar's open
    mark to market at close
    risk.on_day        -> stop-loss / trailing / drawdown circuit breaker
    strategy.on_bar    -> new orders
    fill_at=close     -> match them at this bar's close (optimistic)
    record equity

``next_open`` (default) avoids look-ahead bias: a signal computed from
today's close can only trade at tomorrow's open, as in real life.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .analysis.metrics import compute_metrics, format_metrics
from .core.broker import SimBroker
from .core.context import Context
from .core.rules import MarketRegistry
from .data.base import normalize_bars
from .risk import RiskConfig, RiskManager
from .strategies.base import Strategy

log = logging.getLogger("quantum_analyze")


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.DataFrame                 # datetime -> cash, market_value, total
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict
    benchmark: Optional[pd.Series] = None
    logs: list[str] = field(default_factory=list)
    risk_events: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"策略: {self.strategy}  参数: {self.params}"]
        lines += [f"  {k:<10s} {v}" for k, v in format_metrics(self.metrics)]
        return "\n".join(lines)

    def save(self, out_dir: str) -> str:
        """Write equity / trades / orders CSV and an HTML report; return report path."""
        from pathlib import Path
        from .analysis.report import write_html_report
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.equity.to_csv(p / "equity.csv")
        self.trades.to_csv(p / "trades.csv", index=False)
        self.orders.to_csv(p / "orders.csv", index=False)
        write_json(p / "metrics.json", {"strategy": self.strategy, "params": self.params,
                                        "metrics": self.metrics,
                                        "formatted": format_metrics(self.metrics)})
        return str(write_html_report(self, p / "report.html"))


class Engine:
    def __init__(self, strategy: Strategy, data: dict[str, pd.DataFrame],
                 account=None, risk: Optional[RiskManager] = None,
                 fill_at: str = "next_open", registry: Optional[MarketRegistry] = None,
                 initial_cash: float = 1_000_000.0):
        if fill_at not in ("next_open", "close"):
            raise ValueError("fill_at 必须是 next_open 或 close")
        self.data = {s: (df if "pre_close" in df.columns else normalize_bars(df)) for s, df in data.items()}
        self.strategy = strategy
        self.account = account if account is not None else SimBroker(initial_cash, registry)
        self.risk = risk if risk is not None else RiskManager()
        self.fill_at = fill_at
        self.calendar = _calendar(self.data)
        self.ctx = Context(self.data, self.account, self.risk, self.calendar)
        self.equity_rows: list[tuple] = []
        self._last_day = None
        self._prepared = False

    def prepare(self) -> None:
        if not self._prepared:
            self.strategy.prepare(self.data)
            self._prepared = True

    def step(self, dt: pd.Timestamp) -> None:
        ctx, acct = self.ctx, self.account
        day = dt.normalize()
        new_day = day != self._last_day
        self._last_day = day
        is_sim = isinstance(acct, SimBroker)
        if is_sim:
            acct.on_day_start(dt) if new_day else acct.set_time(dt)
        ctx.set_dt(dt)
        bars = {}
        if is_sim:
            for sym in self.data:
                b = ctx.bar(sym)
                if b is not None:
                    bars[sym] = b
            if self.fill_at == "next_open" and acct.pending:
                acct.match(bars, "open")
            acct.update_prices(bars)
        self.risk.on_day(ctx, new_day)
        self.strategy.on_bar(ctx)
        if is_sim and self.fill_at == "close" and acct.pending:
            acct.match(bars, "close")
            acct.update_prices(bars)
        self.equity_rows.append((dt, acct.cash, acct.total_value - acct.cash, acct.total_value))

    def run(self, start: Optional[str] = None, end: Optional[str] = None,
            benchmark: Optional[pd.Series] = None) -> BacktestResult:
        self.prepare()
        cal = self.calendar
        if start:
            cal = cal[cal >= pd.Timestamp(start)]
        if end:
            cal = cal[cal <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
        if len(cal) == 0:
            raise ValueError("回测区间内没有任何行情数据")
        self.strategy.on_start(self.ctx)
        for dt in cal:
            self.step(dt)
        self.strategy.on_end(self.ctx)
        return self.result(benchmark)

    def result(self, benchmark: Optional[pd.Series] = None) -> BacktestResult:
        eq = pd.DataFrame(self.equity_rows, columns=["datetime", "cash", "market_value", "total"]
                          ).set_index("datetime")
        acct = self.account
        reg = acct.registry
        trades = pd.DataFrame([t.to_dict() for t in getattr(acct, "trades", [])],
                              columns=["trade_id", "order_id", "symbol", "direction", "quantity",
                                       "price", "commission", "tax", "datetime", "pnl", "tag"])
        if len(trades):
            trades["fx"] = trades["symbol"].map(reg.fx)
        orders = pd.DataFrame([o.to_dict() for o in getattr(acct, "orders", [])])
        if benchmark is not None and len(eq):
            benchmark = benchmark[(benchmark.index >= eq.index[0].normalize())
                                  & (benchmark.index <= eq.index[-1])]
        metrics = compute_metrics(eq["total"], trades, benchmark) if len(eq) else {}
        return BacktestResult(self.strategy.name, eq, trades, orders, metrics, benchmark,
                              self.ctx.logs, self.risk.events, dict(self.strategy.params))


def write_json(path, obj) -> None:
    """JSON without NaN/Infinity (browsers reject them)."""
    import json
    import math

    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if hasattr(v, "item"):
            return clean(v.item())
        return v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1, default=str)


def _calendar(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex([])
    for df in data.values():
        idx = idx.union(df.index)
    return idx.sort_values()


def run_backtest(strategy: Strategy, data: dict[str, pd.DataFrame], start: Optional[str] = None,
                 end: Optional[str] = None, initial_cash: float = 1_000_000.0,
                 fill_at: str = "next_open", risk: Optional[RiskConfig] = None,
                 registry: Optional[MarketRegistry] = None,
                 benchmark: Optional[pd.Series] = None) -> BacktestResult:
    """One-call backtest helper."""
    engine = Engine(strategy, data, risk=RiskManager(risk), fill_at=fill_at,
                    registry=registry, initial_cash=initial_cash)
    return engine.run(start, end, benchmark)
