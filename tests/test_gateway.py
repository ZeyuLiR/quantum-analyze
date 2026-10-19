"""Offline tests for broker gateways (no SDKs / broker clients required)."""
from __future__ import annotations

import csv
import json
import sys

import pandas as pd
import pytest

from quantum_analyze.core.objects import Bar, Direction, Order, OrderStatus
from quantum_analyze.core.rules import MarketRegistry
from quantum_analyze.gateway.base import GatewayError, GatewayUnavailable, LiveAccount
from quantum_analyze.gateway import brokers as B


# ---------------------------------------------------------------- symbols
def test_split_symbol():
    assert B.split_symbol("700.hk") == ("00700", "HK")
    assert B.split_symbol("600519") == ("600519", "SH")
    assert B.split_symbol("600519.SS") == ("600519", "SH")
    assert B.split_symbol("BRK.B.US") == ("BRK.B", "US")
    assert B.split_symbol("BRK.B") == ("BRK.B", "US")
    assert B.split_symbol("aapl") == ("AAPL", "US")


@pytest.mark.parametrize("sym, ib", [
    ("AAPL.US", ("AAPL", "SMART", "USD")),
    ("BRK.B.US", ("BRK B", "SMART", "USD")),
    ("00700.HK", ("700", "SEHK", "HKD")),
    ("09988.HK", ("9988", "SEHK", "HKD")),
    ("600519.SH", ("600519", "SEHKNTL", "CNH")),
    ("000001.SZ", ("000001", "SEHKSZSE", "CNH")),
])
def test_ib_roundtrip(sym, ib):
    assert B.to_ib(sym) == ib
    assert B.from_ib(ib[0], ib[2], ib[1]) == sym


def test_ib_from_positions_without_exchange():
    assert B.from_ib("700", "HKD") == "00700.HK"
    assert B.from_ib("600519", "CNH") == "600519.SH"
    assert B.from_ib("000001", "CNH") == "000001.SZ"
    assert B.from_ib("AAPL", "USD", "NASDAQ") == "AAPL.US"
    with pytest.raises(ValueError):
        B.to_ib("430047.BJ")


def test_ib_paper_inference():
    assert B.IBGateway._infer_paper(7496, "DU1234567") is True
    assert B.IBGateway._infer_paper(7497, "U1234567") is False
    assert B.IBGateway._infer_paper(4002) is True
    assert B.IBGateway._infer_paper(4001) is False
    assert B.IBGateway._infer_paper(4001, "U1", explicit=True) is True
    assert B.IBGateway(port=7496).paper is False


@pytest.mark.parametrize("sym, code", [
    ("00700.HK", "HK.00700"), ("AAPL.US", "US.AAPL"), ("BRK.B.US", "US.BRK.B"),
    ("600519.SH", "SH.600519"), ("000001.SZ", "SZ.000001"),
])
def test_futu_roundtrip(sym, code):
    assert B.to_futu(sym) == code
    assert B.from_futu(code) == sym


def test_tiger():
    assert B.to_tiger("00700.HK") == ("00700", "HKD")
    assert B.to_tiger("AAPL.US") == ("AAPL", "USD")
    assert B.from_tiger("00700", "HKD") == "00700.HK"
    assert B.from_tiger("700") == "00700.HK"
    assert B.from_tiger("AAPL", "USD") == "AAPL.US"
    with pytest.raises(ValueError):
        B.to_tiger("600519.SH")


@pytest.mark.parametrize("sym, code", [
    ("00700.HK", "700.HK"), ("AAPL.US", "AAPL.US"), ("600519.SH", "600519.SH"),
    ("000001.SZ", "000001.SZ"),
])
def test_longport_roundtrip(sym, code):
    assert B.to_longport(sym) == code
    assert B.from_longport(code) == sym


def test_qmt_easytrader_ptrade():
    assert B.to_qmt("600519") == "600519.SH"
    assert B.to_qmt("430047.BJ") == "430047.BJ"
    assert B.from_qmt("000001.SZ") == "000001.SZ"
    with pytest.raises(ValueError):
        B.to_qmt("00700.HK")
    assert B.to_easytrader("600519.SH") == "600519"
    assert B.from_easytrader(1) == "000001.SZ"
    assert B.from_easytrader("600519") == "600519.SH"
    assert B.to_ptrade("600519.SH") == "600519.SS"
    assert B.to_ptrade("000001.SZ") == "000001.SZ"
    assert B.from_ptrade("600519.SS") == "600519.SH"


def test_round_price():
    assert B.round_price("00700.HK", 401.37) == 401.4
    assert B.round_price("00700.HK", 15.03) == 15.04
    assert B.round_price("00700.HK", 55.02) == 55.0
    assert B.round_price("AAPL.US", 190.123) == 190.12
    assert B.round_price("AAPL.US", 0.12346) == 0.1235
    assert B.round_price("600519.SH", 1500.006) == 1500.01
    assert B.round_price("510300.SH", 3.9876) == 3.988


def test_combine_balances():
    reg = MarketRegistry(fx_rates={"HKD": 0.92, "USD": 7.2})
    s = B.combine_balances([(100, 1000, "USD"), (1000, 2000, "HKD")], reg)
    assert s.currency == "CNY"
    assert s.cash == pytest.approx(720 + 920)
    assert s.total_value == pytest.approx(7200 + 1840)
    s = B.combine_balances([(1, 2, "USD")])
    assert (s.cash, s.total_value, s.currency) == (1, 2, "USD")
    assert B.combine_balances([(1, 2, "CNH")], reg).currency == "CNY"
    with pytest.raises(GatewayError):
        B.combine_balances([(1, 2, "USD"), (1, 2, "HKD")])


# ---------------------------------------------------------------- factory
def test_registry_and_factory(tmp_path):
    assert set(B.GATEWAYS) == {"ibkr", "futu", "tiger", "longbridge", "qmt", "easytrader",
                               "ptrade", "manual", "sim"}
    assert isinstance(B.build_gateway("sim", initial_cash=10), B.SimGateway)
    assert isinstance(B.build_gateway("IB", port=4002), B.IBGateway)
    assert isinstance(B.build_gateway("longport"), B.LongbridgeGateway)
    with pytest.raises(GatewayError):
        B.build_gateway("nope")


# ---------------------------------------------------------------- SimGateway
def _bar(sym, dt, px, pre=None):
    return Bar(sym, pd.Timestamp(dt), px, px * 1.01, px * 0.99, px, 1e7,
               pre_close=pre if pre is not None else px)


def test_sim_gateway_live_account_end_to_end():
    reg = MarketRegistry(fx_rates={"HKD": 0.92, "USD": 7.2})
    with B.SimGateway(initial_cash=1_000_000, registry=reg) as gw:
        d1 = pd.Timestamp("2026-01-05")
        gw.on_day_start(d1)
        acct = LiveAccount(gw, registry=reg, dry_run=False)
        assert acct.cash == 1_000_000
        o1 = acct.submit(Order("600519.SH", Direction.BUY, 100))
        o2 = acct.submit(Order("00700.HK", Direction.BUY, 200, price=500.0))
        assert o1.status == OrderStatus.SUBMITTED
        assert {o["order_id"] for o in gw.query_open_orders()} == {o1.order_id, o2.order_id}

        trades = gw.match({"600519.SH": _bar("600519.SH", d1, 1500),
                           "00700.HK": _bar("00700.HK", d1, 400)})
        assert len(trades) == 2
        assert o1.status == OrderStatus.FILLED and o1.filled_qty == 100
        assert o2.status == OrderStatus.FILLED
        acct.refresh()
        assert acct.position("600519.SH").quantity == 100
        assert acct.position("600519.SH").sellable == 0            # A-share T+1
        assert acct.position("00700.HK").sellable == 200           # HK T+0
        assert acct.cash < 1_000_000 - 1500 * 100 - 400 * 200 * 0.92
        assert acct.total_value == pytest.approx(gw.broker.total_value)

        rej = acct.submit(Order("600519.SH", Direction.SELL, 100))
        assert rej.status == OrderStatus.REJECTED

        d2 = pd.Timestamp("2026-01-06")
        gw.on_day_start(d2)
        acct.refresh()
        sell = acct.submit(Order("600519.SH", Direction.SELL, 100))
        assert sell.status == OrderStatus.SUBMITTED
        gw.match({"600519.SH": _bar("600519.SH", d2, 1550)})
        assert sell.status == OrderStatus.FILLED
        acct.refresh()
        assert acct.position("600519.SH").quantity == 0


def test_sim_gateway_cancel_and_persist(tmp_path):
    state = tmp_path / "sim.json"
    gw = B.build_gateway("sim", initial_cash=100_000, state_path=str(state))
    gw.connect()
    acct = LiveAccount(gw, dry_run=False)
    o = acct.submit(Order("000001.SZ", Direction.BUY, 100, price=10.0))
    gw.cancel_order(o.order_id)
    assert o.status == OrderStatus.CANCELLED and not gw.broker.pending
    acct.submit(Order("000001.SZ", Direction.BUY, 100))
    gw.match({"000001.SZ": _bar("000001.SZ", "2026-01-05", 10.0)})
    gw.close()
    gw2 = B.SimGateway(state_path=str(state))
    gw2.connect()
    assert gw2.query_positions()["000001.SZ"].quantity == 100
    assert gw2.query_account().cash == pytest.approx(gw.broker.cash)


def test_live_account_dry_run_does_not_send():
    gw = B.SimGateway(initial_cash=100_000)
    gw.connect()
    acct = LiveAccount(gw, dry_run=True)
    o = acct.submit(Order("600000.SH", Direction.BUY, 100))
    assert o in acct.orders and o in acct.pending
    assert "dry-run" in o.reason
    assert gw.broker.pending == [] and gw.broker.orders == []
    acct.cancel_all()
    assert o.status == OrderStatus.CANCELLED


def test_live_account_records_send_failure():
    gw = B.SimGateway(initial_cash=100_000)
    gw.connect()
    acct = LiveAccount(gw, dry_run=False)
    o = acct.submit(Order("600000.SH", Direction.BUY, 0))
    assert o.status == OrderStatus.REJECTED and "发送失败" in o.reason


# ---------------------------------------------------------------- Manual / Ptrade
def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_manual_gateway_writes_csv(tmp_path):
    state = tmp_path / "account.json"
    state.write_text(json.dumps({
        "cash": 50_000, "currency": "CNY",
        "positions": [{"symbol": "600519.SH", "quantity": 100, "avg_cost": 1500,
                       "last_price": 1520, "today_bought": 100},
                      {"symbol": "700.HK", "quantity": 200, "avg_cost": 380, "last_price": 400}],
    }), encoding="utf-8")
    orders = tmp_path / "out" / "orders.csv"
    reg = MarketRegistry(fx_rates={"HKD": 0.9})
    gw = B.ManualGateway(orders_path=str(orders), state_path=str(state), registry=reg)
    gw.connect()
    acct = LiveAccount(gw, registry=reg, dry_run=False)
    assert acct.cash == 50_000
    assert acct.total_value == pytest.approx(50_000 + 152_000 + 200 * 400 * 0.9)
    assert acct.position("00700.HK").quantity == 200
    assert acct.position("600519.SH").sellable == 0
    assert acct.submit(Order("600519.SH", Direction.SELL, 100)).status == OrderStatus.REJECTED

    o = acct.submit(Order("00700.HK", Direction.SELL, 100, price=401.0, tag="rebal"))
    acct.submit(Order("AAPL.US", Direction.BUY, 5))
    rows = _rows(orders)
    assert [r["order_id"] for r in rows] == ["M000001", "M000002"]
    assert rows[0]["symbol"] == "00700.HK" and rows[0]["direction"] == "sell"
    assert rows[0]["quantity"] == "100" and rows[0]["price"] == "401.0" and rows[0]["tag"] == "rebal"
    assert rows[1]["price"] == ""
    acct.cancel_all("00700.HK")
    rows = _rows(orders)
    assert rows[-1]["action"] == "cancel" and rows[-1]["order_id"] == o.order_id
    assert [x["order_id"] for x in gw.query_open_orders()] == ["M000002"]

    gw2 = B.ManualGateway(orders_path=str(orders))   # sequence continues across runs
    gw2.connect()
    assert gw2.send_order(Order("AAPL.US", Direction.BUY, 1)) == "M000003"


def test_manual_gateway_csv_state(tmp_path):
    state = tmp_path / "pos.csv"
    state.write_text("symbol,quantity,avg_cost,last_price\nCASH,1000,,\n000001.SZ,300,10,11\n",
                     encoding="utf-8")
    gw = B.ManualGateway(orders_path=str(tmp_path / "o.csv"), state_path=str(state))
    gw.connect()
    snap = gw.query_account()
    assert snap.cash == 1000 and snap.total_value == pytest.approx(1000 + 3300)
    assert gw.query_positions()["000001.SZ"].sellable == 300


def test_ptrade_export(tmp_path):
    gw = B.build_gateway("ptrade", orders_path=str(tmp_path / "p.csv"))
    gw.connect()
    gw.send_order(Order("600519.SH", Direction.BUY, 100, price=1500.0))
    assert _rows(tmp_path / "p.csv")[0]["code"] == "600519.SS"
    p = B.export_ptrade_orders([Order("000001.SZ", Direction.SELL, 200)], tmp_path / "x.csv")
    r = _rows(p)
    assert len(r) == 1 and r[0]["code"] == "000001.SZ" and r[0]["direction"] == "sell"
    assert "get_research_path" in B.PTRADE_STRATEGY_TEMPLATE
    compile(B.PTRADE_STRATEGY_TEMPLATE, "ptrade", "exec")


# ---------------------------------------------------------------- missing SDKs
@pytest.mark.parametrize("cls, kwargs, modules", [
    (B.IBGateway, {}, ["ib_insync", "ib_async"]),
    (B.FutuGateway, {}, ["futu"]),
    (B.TigerGateway, {"props_path": "x"}, ["tigeropen"]),
    (B.LongbridgeGateway, {}, ["longport", "longbridge"]),
    (B.QMTGateway, {"path": "x", "account_id": "1"}, ["xtquant"]),
    (B.EasyTraderGateway, {"exe_path": "x"}, ["easytrader"]),
])
def test_missing_sdk_raises_unavailable(monkeypatch, cls, kwargs, modules):
    for m in list(sys.modules):
        if any(m == x or m.startswith(x + ".") for x in modules):
            monkeypatch.delitem(sys.modules, m)
    for m in modules:
        monkeypatch.setitem(sys.modules, m, None)    # makes `import m` raise ImportError
    gw = cls(**kwargs)
    with pytest.raises(GatewayUnavailable) as ei:
        gw.connect()
    assert "pip install" in str(ei.value)


# ---------------------------------------------------------------- fake SDKs (code-path smoke tests)
from types import ModuleType, SimpleNamespace as NS


def test_ib_gateway_with_fake_sdk(monkeypatch):
    placed = []

    class Stock(NS):
        def __init__(self, symbol, exchange, currency):
            super().__init__(symbol=symbol, exchange=exchange, currency=currency, conId=0,
                             secType="STK", primaryExchange="")

    def _order(kind):
        def make(action, qty, *price, **kw):
            return NS(orderType=kind, action=action, totalQuantity=qty,
                      lmtPrice=price[0] if price else 0.0, orderId=len(placed) + 1, permId=0, **kw)
        return make

    class IB:
        def connect(self, host, port, clientId, timeout, readonly, account):
            self.connected = True

        def isConnected(self):
            return True

        def disconnect(self):
            self.connected = False

        def managedAccounts(self):
            return ["DU111"]

        def accountSummary(self, account):
            return [NS(account="DU111", tag="NetLiquidation", value="10000", currency="USD"),
                    NS(account="DU111", tag="TotalCashValue", value="4000", currency="USD"),
                    NS(account="DU111", tag="AvailableFunds", value="3500", currency="USD")]

        def portfolio(self):
            return [NS(account="DU111", contract=NS(conId=7), marketPrice=410.0)]

        def positions(self, account):
            c = NS(symbol="700", currency="HKD", exchange="", primaryExchange="SEHK",
                   secType="STK", conId=7)
            return [NS(contract=c, position=200.0, avgCost=380.5)]

        def qualifyContracts(self, *cs):
            for c in cs:
                c.conId = 99
            return list(cs)

        def placeOrder(self, contract, order):
            t = NS(contract=contract, order=order, orderStatus=NS(status="Submitted", filled=0), log=[])
            placed.append(t)
            return t

        def sleep(self, s):
            pass

        def openTrades(self):
            return list(placed)

        def trades(self):
            return list(placed)

        def cancelOrder(self, order):
            order.cancelled = True

    fake = ModuleType("ib_async")
    fake.IB, fake.Stock = IB, Stock
    fake.MarketOrder, fake.LimitOrder = _order("MKT"), _order("LMT")
    monkeypatch.setitem(sys.modules, "ib_async", fake)

    reg = MarketRegistry(fx_rates={"USD": 7.0, "HKD": 0.9})
    gw = B.IBGateway(port=7496, registry=reg)
    gw.connect()
    assert gw.account == "DU111" and gw.paper is True
    snap = gw.query_account()
    assert (snap.cash, snap.total_value, snap.currency) == (3500 * 7.0, 10000 * 7.0, "CNY")
    pos = gw.query_positions()["00700.HK"]
    assert pos.quantity == 200 and pos.last_price == 410.0 and pos.sellable == 200
    oid = gw.send_order(Order("00700.HK", Direction.BUY, 100, price=401.37))
    assert placed[-1].order.lmtPrice == 401.4 and placed[-1].order.tif == "DAY"
    assert placed[-1].contract.symbol == "700" and placed[-1].order.account == "DU111"
    gw.send_order(Order("AAPL.US", Direction.SELL, 3))
    assert placed[-1].order.orderType == "MKT"
    with pytest.raises(GatewayError):
        gw.send_order(Order("600519.SH", Direction.BUY, 100))   # Stock Connect: no market orders
    gw.cancel_order(oid)
    assert placed[0].order.cancelled
    assert gw.query_open_orders()[0]["symbol"] == "00700.HK"
    gw.close()


def test_qmt_gateway_with_fake_sdk(monkeypatch):
    calls = []
    xc = NS(STOCK_BUY=23, STOCK_SELL=24, FIX_PRICE=11, MARKET_SH_CONVERT_5_CANCEL=42,
            MARKET_SZ_CONVERT_5_CANCEL=47)

    class Trader:
        def __init__(self, path, session):
            self.path = path

        def start(self):
            pass

        def connect(self):
            return 0

        def subscribe(self, acc):
            return 0

        def stop(self):
            pass

        def query_stock_asset(self, acc):
            return NS(cash=5000.0, total_asset=20000.0)

        def query_stock_positions(self, acc):
            return [NS(stock_code="600519.SH", volume=100, can_use_volume=0, avg_price=1490.0,
                       open_price=1490.0, market_value=150000.0)]

        def order_stock(self, *a):
            calls.append(a)
            return 1234

        def cancel_order_stock(self, acc, oid):
            return 0

    pkg = ModuleType("xtquant")
    pkg.xtconstant = xc
    pkg.xtdata = NS(get_full_tick=lambda codes: {c: {"lastPrice": 10.0} for c in codes})
    trader_mod = ModuleType("xtquant.xttrader")
    trader_mod.XtQuantTrader = Trader
    type_mod = ModuleType("xtquant.xttype")
    type_mod.StockAccount = lambda acc_id: NS(account_id=acc_id)
    for name, mod in [("xtquant", pkg), ("xtquant.xttrader", trader_mod), ("xtquant.xttype", type_mod),
                      ("xtquant.xtconstant", xc), ("xtquant.xtdata", pkg.xtdata)]:
        monkeypatch.setitem(sys.modules, name, mod)

    gw = B.QMTGateway(path="userdata_mini", account_id="888")
    gw.connect()
    assert gw.query_account().total_value == 20000.0
    p = gw.query_positions()["600519.SH"]
    assert p.today_bought == 100 and p.last_price == 1500.0
    gw.send_order(Order("000001.SZ", Direction.BUY, 100))
    assert calls[-1][1:6] == ("000001.SZ", 23, 100, 47, 10.2)
    gw.send_order(Order("430047.BJ", Direction.SELL, 100))
    assert calls[-1][3:6] == (100, 11, 9.8)
    gw.send_order(Order("600519.SH", Direction.SELL, 100, price=1500.0))
    assert calls[-1][2:6] == (24, 100, 11, 1500.0)
    gw.cancel_order("1234")
    gw.close()


def test_futu_gateway_with_fake_sdk(monkeypatch):
    sent = []

    class Ctx:
        def __init__(self, filter_trdmarket, host, port, security_firm):
            self.m = filter_trdmarket

        def get_acc_list(self):
            return 0, pd.DataFrame({"acc_id": [1, 2], "trd_env": ["REAL", "SIMULATE"],
                                    "sim_acc_type": ["N/A", "STOCK"]})

        def accinfo_query(self, **kw):
            return 0, pd.DataFrame({"cash": [1000.0], "total_assets": [3000.0]})

        def position_list_query(self, **kw):
            code = "HK.00700" if self.m == "HK" else "US.AAPL"
            return 0, pd.DataFrame({"code": [code], "qty": [100.0], "can_sell_qty": [100.0],
                                    "cost_price": [10.0], "nominal_price": [11.0],
                                    "position_side": ["LONG"]})

        def place_order(self, **kw):
            sent.append(kw)
            return 0, pd.DataFrame({"order_id": ["555"]})

        def modify_order(self, op, oid, qty, price, **kw):
            return 0, None

        def close(self):
            pass

    E = lambda *names: NS(**{n: n for n in names})
    ft = ModuleType("futu")
    ft.RET_OK = 0
    ft.OpenSecTradeContext = Ctx
    ft.TrdMarket, ft.SecurityFirm = E("HK", "US"), E("FUTUSECURITIES")
    ft.TrdEnv, ft.TrdSide = E("REAL", "SIMULATE"), E("BUY", "SELL")
    ft.OrderType, ft.ModifyOrderOp = E("MARKET", "NORMAL"), E("CANCEL")
    monkeypatch.setitem(sys.modules, "futu", ft)

    reg = MarketRegistry(fx_rates={"HKD": 0.9, "USD": 7.0})
    gw = B.FutuGateway(paper=True, registry=reg)
    gw.connect()
    assert gw._acc == {"HK": 2, "US": 2}
    snap = gw.query_account()                        # same acc id -> counted once
    assert snap.total_value == pytest.approx(3000 * 0.9)
    assert set(gw.query_positions()) == {"00700.HK", "AAPL.US"}
    assert gw.send_order(Order("00700.HK", Direction.BUY, 100)) == "555"
    assert sent[-1]["code"] == "HK.00700" and sent[-1]["order_type"] == "MARKET"
    assert sent[-1]["trd_env"] == "SIMULATE"
    gw.cancel_order("555")
    gw.close()
