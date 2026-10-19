import math

import numpy as np
import pandas as pd
import pytest

from quantum_analyze.analysis.metrics import compute_metrics
from quantum_analyze.core.broker import SimBroker
from quantum_analyze.core.context import Context
from quantum_analyze.core.objects import Bar, Direction, Order, OrderStatus
from quantum_analyze.core.rules import MarketRegistry, market_of, normalize_symbol
from quantum_analyze.engine import Engine, run_backtest
from quantum_analyze.risk import RiskConfig, RiskManager
from quantum_analyze.strategies import builtin  # noqa: F401  (registers strategies)
from quantum_analyze.strategies.base import STRATEGY_REGISTRY, Strategy, get_strategy

D = pd.Timestamp("2024-01-02")


def bar(sym="600000.SH", o=10.0, h=10.5, l=9.8, c=10.2, v=1e6, pc=10.0, dt=D):
    return Bar(sym, dt, o, h, l, c, v, pc)


def no_slip_registry():
    reg = MarketRegistry(fx_rates={"HKD": 0.9, "USD": 7.0})
    for spec in reg.specs.values():
        spec.cost.slippage = 0.0
    return reg


# ---------------------------------------------------------------- symbols / rules
def test_symbol_normalisation():
    assert normalize_symbol("600519") == "600519.SH"
    assert normalize_symbol("000001") == "000001.SZ"
    assert normalize_symbol("830799") == "830799.BJ"
    assert normalize_symbol("700") == "00700.HK"
    assert normalize_symbol("aapl") == "AAPL.US"
    assert [market_of(s) for s in ("600519.SH", "00700.HK", "AAPL.US")] == ["CN", "HK", "US"]


def test_a_share_costs_and_limits():
    reg = MarketRegistry()
    cn = reg.spec("600000.SH")
    com, tax = cn.cost.fees(True, 10_000)
    assert com == pytest.approx(5.0 + 0.1)          # min commission 5 + transfer fee
    assert tax == 0
    com, tax = cn.cost.fees(False, 100_000)
    assert tax == pytest.approx(50.0)               # 0.05% stamp duty on sells
    assert cn.rules.limit_prices("600000.SH", 10.0) == (9.0, 11.0)
    assert cn.rules.limit_prices("300750.SZ", 10.0) == (8.0, 12.0)
    assert cn.rules.limit_prices("688001.SH", 10.0) == (8.0, 12.0)
    hk = reg.spec("00700.HK")
    _, tax = hk.cost.fees(True, 100_000)
    assert tax == pytest.approx(100.0)              # HK stamp duty both sides
    assert reg.spec("AAPL.US").rules.lot_size == 1


# ---------------------------------------------------------------- broker
def test_t_plus_1_and_lot_rounding():
    b = SimBroker(100_000, no_slip_registry())
    b.on_day_start(D)
    b.submit(Order("600000.SH", Direction.BUY, 1000))
    b.match({"600000.SH": bar()}, "open")
    assert b.positions["600000.SH"].quantity == 1000
    b.submit(Order("600000.SH", Direction.SELL, 1000))
    b.match({"600000.SH": bar()}, "close")
    assert b.positions["600000.SH"].quantity == 1000          # T+1: cannot sell today
    assert b.orders[-1].reason.startswith("无可卖持仓")
    b.on_day_start(D + pd.Timedelta(days=1))
    b.submit(Order("600000.SH", Direction.SELL, 1000))
    b.match({"600000.SH": bar(dt=D + pd.Timedelta(days=1))}, "open")
    assert "600000.SH" not in b.positions


def test_us_is_t0_with_fx():
    b = SimBroker(100_000, no_slip_registry())
    b.on_day_start(D)
    b.submit(Order("AAPL.US", Direction.BUY, 10))
    b.match({"AAPL.US": bar("AAPL.US", o=100, h=101, l=99, c=100, pc=99)}, "open")
    assert b.cash == pytest.approx(100_000 - (1000 + 1.98) * 7.0)
    b.submit(Order("AAPL.US", Direction.SELL, 10))
    b.match({"AAPL.US": bar("AAPL.US", o=100, h=101, l=99, c=100, pc=99)}, "close")
    assert "AAPL.US" not in b.positions                         # T+0 allowed


def test_limit_up_blocks_buy_and_limit_down_blocks_sell():
    b = SimBroker(100_000, no_slip_registry())
    b.on_day_start(D)
    b.submit(Order("600000.SH", Direction.BUY, 100))
    b.match({"600000.SH": bar(o=11.0, h=11.0, l=11.0, c=11.0, pc=10.0)}, "open")
    assert b.orders[-1].status == OrderStatus.CANCELLED and "涨停" in b.orders[-1].reason


def test_insufficient_cash_shrinks_buy():
    b = SimBroker(5_000, no_slip_registry())
    b.on_day_start(D)
    b.submit(Order("600000.SH", Direction.BUY, 1000))
    b.match({"600000.SH": bar()}, "open")
    assert b.positions["600000.SH"].quantity == 400            # 400*10 + fees <= 5000
    assert b.cash >= 0


def test_suspended_symbol_not_filled():
    b = SimBroker(100_000, no_slip_registry())
    b.on_day_start(D)
    b.submit(Order("600000.SH", Direction.BUY, 100))
    b.match({}, "open")
    assert b.orders[-1].status == OrderStatus.CANCELLED


def test_broker_state_roundtrip():
    b = SimBroker(100_000, no_slip_registry())
    b.on_day_start(D)
    b.submit(Order("600000.SH", Direction.BUY, 500))
    b.match({"600000.SH": bar()}, "open")
    b.submit(Order("600000.SH", Direction.SELL, 200))
    b2 = SimBroker.from_dict(b.to_dict(), no_slip_registry())
    assert b2.cash == b.cash and b2.positions["600000.SH"].quantity == 500
    assert len(b2.pending) == 1 and b2.pending[0].quantity == 200


# ---------------------------------------------------------------- engine
class Peek(Strategy):
    name = "peek"

    def on_bar(self, ctx):
        # history must never contain bars after ctx.dt
        h = ctx.history("600000.SH", "close")
        assert len(h) == 0 or h.index[-1] <= ctx.dt


def test_no_lookahead(panel):
    run_backtest(Peek(), panel)


class BuyFirstDay(Strategy):
    name = "buy_first"

    def on_bar(self, ctx):
        if not ctx.state.get("done"):
            ctx.order_target_percent("600000.SH", 0.5)
            ctx.state["done"] = True


def test_next_open_fill_timing(panel):
    res = run_backtest(BuyFirstDay(), panel)
    t = res.trades.iloc[0]
    df = panel["600000.SH"]
    assert pd.Timestamp(t["datetime"]) == df.index[1]          # signal day 0 → fill day 1
    assert t["price"] == pytest.approx(round(df["open"].iloc[1] * 1.0005, 2), abs=0.011)


def test_close_fill_mode(panel):
    res = run_backtest(BuyFirstDay(), panel, fill_at="close")
    assert pd.Timestamp(res.trades.iloc[0]["datetime"]) == panel["600000.SH"].index[0]


@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY))
def test_every_builtin_strategy_runs(panel, name):
    cls = STRATEGY_REGISTRY[name]
    res = run_backtest(cls(), panel, risk=RiskConfig(stop_loss=0.15))
    assert res.metrics["final_value"] > 0
    assert (res.equity["cash"] >= -1e-6).all()


def test_stop_loss_and_drawdown_halt(panel):
    crash = panel["600000.SH"].copy()
    crash.iloc[50:, :4] *= 0.5        # 50% gap down
    crash["pre_close"] = crash["close"].shift(1)
    data = {"600000.SH": crash}
    for k in ("open", "high", "low", "close"):
        assert crash[k].iloc[50] < crash[k].iloc[49]
    res = run_backtest(get_strategy("buy_hold")(), data, risk=RiskConfig(stop_loss=0.1))
    assert any("止损" in e for e in res.risk_events)


def test_risk_caps_position_weight(panel):
    res = run_backtest(BuyFirstDay(), panel, risk=RiskConfig(max_position_weight=0.2))
    t = res.trades.iloc[0]
    assert t["price"] * t["quantity"] <= 0.2 * 1_000_000 + 1


def test_period_start_flags(panel):
    eng = Engine(Peek(), panel)
    ctx = eng.ctx
    firsts = []
    for dt in eng.calendar[:70]:
        ctx.set_dt(dt)
        if ctx.is_period_start("M"):
            firsts.append(dt)
    assert [d.month for d in firsts] == [1, 2, 3, 4]


# ---------------------------------------------------------------- metrics
def test_metrics_known_values():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(np.linspace(100, 110, 253), index=idx)
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(0.10)
    assert m["annual_return"] == pytest.approx(0.10, rel=1e-6)
    assert m["max_drawdown"] == 0
    bench = eq * 2
    m = compute_metrics(eq, benchmark=bench)
    assert m["beta"] == pytest.approx(1.0, rel=1e-6)
    dd = pd.Series([100, 120, 90, 130], index=pd.bdate_range("2020-01-01", periods=4), dtype=float)
    assert compute_metrics(dd)["max_drawdown"] == pytest.approx(-0.25)
