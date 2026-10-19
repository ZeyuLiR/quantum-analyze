import json

import pandas as pd
import pytest

from quantum_analyze.config import AppConfig
from quantum_analyze.optimize import grid_search, walk_forward
from quantum_analyze.risk import RiskConfig, RiskManager
from quantum_analyze.strategies.base import get_strategy
from quantum_analyze.trader import PaperTrader

from .conftest import make_bars


def test_grid_search_skips_invalid_pairs(panel):
    table = grid_search(get_strategy("dual_ma"), panel, {"fast": [5, 30], "slow": [20, 60]})
    assert len(table) == 3                       # (30, 20) is invalid
    assert table["sharpe"].is_monotonic_decreasing


def test_walk_forward_is_out_of_sample():
    data = {f"{600000 + i:06d}.SH": make_bars(i, n=1000, start="2018-01-01") for i in range(4)}
    out = walk_forward(get_strategy("momentum_rotation"), data, {"lookback": [20, 60]},
                       train_months=12, test_months=6)
    w = out["windows"]
    assert len(w) >= 3
    for _, row in w.iterrows():
        train_end = pd.Timestamp(row["train"].split("~")[1])
        test_start = pd.Timestamp(row["test"].split("~")[0])
        assert test_start > train_end
    assert out["oos_metrics"]["final_value"] > 0


def _cfg(tmp_path, **trading):
    p = tmp_path / "cfg.toml"
    p.write_text("""
[account]
initial_cash = 500000
[universe]
symbols = ["600000", "600001", "600002"]
[strategy]
name = "momentum_rotation"
[strategy.params]
lookback = 20
top_k = 1
rebalance = "D"
[risk]
stop_loss = 0.2
""", encoding="utf-8")
    return AppConfig.load(p)


def test_paper_trader_incremental_runs(tmp_path, panel):
    cfg = _cfg(tmp_path)
    data = {k: v for k, v in panel.items() if k in cfg.symbols}
    dates = next(iter(data.values())).index
    trader = PaperTrader(cfg)
    first = trader.run(data=data, until=str(dates[300].date()))
    assert first["bars_processed"] == 1 and first["next_orders"]
    second = trader.run(data=data, until=str(dates[305].date()))
    assert second["bars_processed"] == 5
    assert second["new_trades"], "yesterday's orders should fill at today's open"
    st = json.loads((tmp_path / "state" / "paper.json").read_text(encoding="utf-8"))
    assert st["last_bar"].startswith(str(dates[305].date()))
    assert len(st["equity"]) == 6
    again = trader.run(data=data, until=str(dates[305].date()))
    assert again["bars_processed"] == 0          # idempotent


def test_paper_matches_backtest(tmp_path, panel):
    """Running paper day-by-day must reproduce the backtest exactly."""
    from quantum_analyze.engine import run_backtest
    cfg = _cfg(tmp_path)
    data = {k: v for k, v in panel.items() if k in cfg.symbols}
    dates = next(iter(data.values())).index
    start = dates[200]
    bt = run_backtest(cfg.build_strategy(), data, start=str(start.date()), end=str(dates[260].date()),
                      initial_cash=cfg.initial_cash, risk=cfg.build_risk(), registry=cfg.build_registry())
    trader = PaperTrader(cfg)
    trader.run(data=data, until=str(start.date()))
    for d in dates[201:261:7]:
        trader.run(data=data, until=str(d.date()))
    trader.run(data=data, until=str(dates[260].date()))
    st = json.loads((tmp_path / "state" / "paper.json").read_text(encoding="utf-8"))
    assert st["equity"][-1][1] == pytest.approx(bt.equity["total"].iloc[-1], rel=1e-9)


def test_config_overrides(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("""
[account]
base_currency = "CNY"
fx_rates = { HKD = 0.92, USD = 7.1 }
[cost.CN]
commission_rate = 0.0001
min_commission = 0.0
[rules.HK]
lot_sizes = { "5" = 400 }
[strategy]
name = "dual_ma"
[strategy.params]
fast = 3
""", encoding="utf-8")
    cfg = AppConfig.load(p)
    reg = cfg.build_registry()
    assert reg.spec("600000.SH").cost.commission_rate == 0.0001
    assert reg.spec("00005.HK").rules.lot_for("00005.HK") == 400
    assert reg.fx("AAPL.US") == 7.1
    assert cfg.build_strategy().params["fast"] == 3
    with pytest.raises(ValueError):
        AppConfig.load(_write(tmp_path, "[nonsense]\na=1\n"))


def _write(tmp_path, text):
    p = tmp_path / "bad.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_drawdown_halt_blocks_buys(panel):
    from quantum_analyze.engine import run_backtest
    df = panel["600000.SH"].copy()
    df.iloc[100:, :4] *= 0.85
    df["pre_close"] = df["close"].shift(1)
    res = run_backtest(get_strategy("dual_ma")(fast=2, slow=5), {"600000.SH": df},
                       risk=RiskConfig(max_drawdown_halt=0.05, halt_cooldown_days=10))
    assert any("熔断" in e for e in res.risk_events)
    rejected = res.orders[res.orders["status"] == "rejected"]
    assert rejected["reason"].str.contains("熔断").any()


def test_live_trader_with_sim_gateway(tmp_path, panel):
    from quantum_analyze.gateway.brokers import SimGateway
    from quantum_analyze.trader import LiveTrader
    cfg = _cfg(tmp_path)
    data = {k: v.iloc[:300] for k, v in panel.items() if k in cfg.symbols}
    gw = SimGateway(initial_cash=500_000, registry=cfg.build_registry())
    dry = LiveTrader(cfg, gateway=gw, dry_run=True).run(data=data)
    assert dry["orders"] and not gw.broker.pending           # dry-run sends nothing
    live = LiveTrader(cfg, gateway=gw, dry_run=False).run(data=data)
    assert len(gw.broker.pending) == len(live["orders"]) >= 1
    again = LiveTrader(cfg, gateway=gw, dry_run=False).run(data=data)
    assert again.get("skipped")                              # same bar is not traded twice
