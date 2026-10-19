"""Concrete broker gateways.

Every gateway imports its broker SDK lazily inside ``connect()`` and raises
``GatewayUnavailable`` with an install / client-setup hint when it is missing,
so the core package only needs numpy + pandas.

=============  ==========================================  ===========  =========
name           broker / client                             markets      SDK
=============  ==========================================  ===========  =========
``ibkr``       Interactive Brokers TWS / IB Gateway        US HK (CN*)  ib_async / ib_insync
``futu``       Futu / moomoo OpenD                         HK US        futu-api
``tiger``      Tiger Brokers Open API                      HK US        tigeropen
``longbridge`` Longbridge (LongPort) OpenAPI               HK US (CN*)  longport
``qmt``        miniQMT (many A-share brokers)              CN           xtquant
``easytrader`` 同花顺 / 通用客户端 GUI automation (Windows)  CN           easytrader
``ptrade``     export orders as CSV for a Ptrade strategy  CN           -
``manual``     orders to CSV, account from a JSON/CSV file any          -
``sim``        built-in local simulated matching           any          -
=============  ==========================================  ===========  =========

``CN*`` = A-shares through Stock Connect northbound (沪深股通), limit orders only.

Symbols are always ``CODE.MARKET`` (``600519.SH``, ``00700.HK``, ``AAPL.US``);
the ``to_xxx`` / ``from_xxx`` helpers convert to and from each broker's codes.
Amounts in ``AccountSnapshot`` are reported in the account's base currency; pass
a ``MarketRegistry`` to convert them into the registry's base currency (CNY).
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import logging
import math
import sys
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from ..core.broker import SimBroker
from ..core.objects import Bar, Direction, Order, OrderStatus, Position, Trade
from ..core.rules import MarketRegistry, normalize_symbol
from .base import AccountSnapshot, BaseGateway, GatewayError, GatewayUnavailable

log = logging.getLogger("quantum_analyze.gateway")

__all__ = [
    "IBGateway", "FutuGateway", "TigerGateway", "LongbridgeGateway", "QMTGateway",
    "EasyTraderGateway", "PtradeGateway", "ManualGateway", "SimGateway",
    "GATEWAYS", "build_gateway", "export_ptrade_orders", "PTRADE_STRATEGY_TEMPLATE",
    "split_symbol", "round_price", "combine_balances",
    "to_ib", "from_ib", "to_futu", "from_futu", "to_tiger", "from_tiger",
    "to_longport", "from_longport", "to_qmt", "from_qmt",
    "to_easytrader", "from_easytrader", "to_ptrade", "from_ptrade",
]

_KNOWN_SUFFIXES = ("SH", "SZ", "BJ", "SS", "HK", "US")


# =========================================================================
# symbol conversion (pure functions)
# =========================================================================
def split_symbol(symbol: str) -> tuple[str, str]:
    """``'700.hk'`` -> ``('00700', 'HK')``; ``'BRK.B.US'`` -> ``('BRK.B', 'US')``.

    The suffix is one of SH / SZ / BJ / HK / US (``.SS`` is read as SH).
    """
    s = normalize_symbol(symbol)
    code, suffix = s.rsplit(".", 1)
    if suffix not in _KNOWN_SUFFIXES:          # e.g. "BRK.B" without ".US"
        return s, "US"
    if suffix == "SS":
        suffix = "SH"
    if suffix == "HK":
        code = code.zfill(5)
    return code, suffix


def _cn_symbol(code: str) -> str:
    return normalize_symbol(str(code).strip().zfill(6))


# ---- Interactive Brokers
def to_ib(symbol: str) -> tuple[str, str, str]:
    """Return ``(symbol, exchange, currency)`` for ``ib.Stock(...)``.

    US -> ``('BRK B', 'SMART', 'USD')``; HK -> ``('700', 'SEHK', 'HKD')``;
    SH/SZ via Stock Connect -> ``('600519', 'SEHKNTL'|'SEHKSZSE', 'CNH')``.
    """
    code, mkt = split_symbol(symbol)
    if mkt == "US":
        return code.replace(".", " "), "SMART", "USD"
    if mkt == "HK":
        return str(int(code)), "SEHK", "HKD"
    if mkt == "SH":
        return code, "SEHKNTL", "CNH"
    if mkt == "SZ":
        return code, "SEHKSZSE", "CNH"
    raise ValueError(f"IBKR 不支持该市场: {symbol}")


def from_ib(symbol: str, currency: str = "", exchange: str = "") -> str:
    """Inverse of :func:`to_ib` from an IB contract's symbol/currency/exchange
    (pass ``primaryExchange`` when ``exchange`` is empty)."""
    ex, cur = (exchange or "").upper(), (currency or "").upper()
    if ex == "SEHKNTL":
        return f"{symbol.zfill(6)}.SH"
    if ex == "SEHKSZSE":
        return f"{symbol.zfill(6)}.SZ"
    if cur in ("CNH", "CNY") and symbol.isdigit():
        return _cn_symbol(symbol)
    if cur == "HKD" or ex == "SEHK":
        return f"{symbol.zfill(5)}.HK"
    return f"{symbol.replace(' ', '.')}.US"


# ---- Futu / moomoo
def to_futu(symbol: str) -> str:
    """``00700.HK`` -> ``HK.00700``; ``AAPL.US`` -> ``US.AAPL``; ``600519.SH`` -> ``SH.600519``."""
    code, mkt = split_symbol(symbol)
    if mkt == "BJ":
        raise ValueError(f"富途不支持北交所: {symbol}")
    return f"{mkt}.{code}"


def from_futu(code: str) -> str:
    mkt, rest = code.split(".", 1)
    mkt = mkt.upper()
    if mkt == "HK":
        return f"{rest.zfill(5)}.HK"
    if mkt in ("SH", "SZ"):
        return f"{rest}.{mkt}"
    return f"{rest}.US"


# ---- Tiger
def to_tiger(symbol: str) -> tuple[str, str]:
    """Return ``(symbol, currency)`` for ``stock_contract``: HK ``('00700', 'HKD')``."""
    code, mkt = split_symbol(symbol)
    if mkt == "HK":
        return code, "HKD"
    if mkt == "US":
        return code, "USD"
    raise ValueError(f"老虎网关暂只支持港股/美股: {symbol}")


def from_tiger(symbol: str, currency: str = "") -> str:
    cur = (currency or "").upper()
    if cur == "HKD" or (symbol.isdigit() and len(symbol) <= 5):
        return f"{symbol.zfill(5)}.HK"
    if cur in ("CNH", "CNY") and symbol.isdigit():
        return _cn_symbol(symbol)
    return f"{symbol}.US"


# ---- Longbridge / LongPort
def to_longport(symbol: str) -> str:
    """``00700.HK`` -> ``700.HK``; ``AAPL.US`` / ``600519.SH`` unchanged."""
    code, mkt = split_symbol(symbol)
    if mkt == "HK":
        return f"{int(code)}.HK"
    if mkt == "BJ":
        raise ValueError(f"长桥不支持北交所: {symbol}")
    return f"{code}.{mkt}"


def from_longport(symbol: str) -> str:
    code, mkt = symbol.rsplit(".", 1)
    mkt = mkt.upper()
    if mkt == "HK":
        return f"{code.zfill(5)}.HK"
    return f"{code}.{mkt}"


# ---- miniQMT
def to_qmt(symbol: str) -> str:
    code, mkt = split_symbol(symbol)
    if mkt not in ("SH", "SZ", "BJ"):
        raise ValueError(f"QMT 仅支持 A 股: {symbol}")
    return f"{code}.{mkt}"


def from_qmt(code: str) -> str:
    c, mkt = split_symbol(code)
    return f"{c}.{mkt}"


# ---- easytrader (GUI clients use bare 6-digit codes)
def to_easytrader(symbol: str) -> str:
    code, mkt = split_symbol(symbol)
    if mkt not in ("SH", "SZ", "BJ"):
        raise ValueError(f"easytrader 仅支持 A 股: {symbol}")
    return code


def from_easytrader(code: Any) -> str:
    return _cn_symbol(str(code).split(".")[0])


# ---- Ptrade (Hundsun): Shanghai is ".SS"
def to_ptrade(symbol: str) -> str:
    code, mkt = split_symbol(symbol)
    if mkt == "SH":
        return f"{code}.SS"
    if mkt in ("SZ", "BJ"):
        return f"{code}.{mkt}"
    raise ValueError(f"Ptrade 仅支持 A 股: {symbol}")


def from_ptrade(code: str) -> str:
    return from_qmt(code)


_CODE_FORMATS = {
    "standard": lambda s: "{}.{}".format(*split_symbol(s)),
    "ptrade": to_ptrade, "qmt": to_qmt, "easytrader": to_easytrader,
    "futu": to_futu, "longport": to_longport, "ib": lambda s: " ".join(to_ib(s)),
}

# ---- price ticks
_HK_TICKS = [(0.25, 0.001), (0.5, 0.005), (10, 0.01), (20, 0.02), (100, 0.05), (200, 0.1),
             (500, 0.2), (1000, 0.5), (2000, 1.0), (5000, 2.0), (math.inf, 5.0)]


def round_price(symbol: str, price: float) -> float:
    """Round a limit price to the market's valid tick (HK spread table,
    US 0.01 / 0.0001 below $1, A-share 0.01 or 0.001 for funds)."""
    code, mkt = split_symbol(symbol)
    if mkt == "HK":
        tick = next(t for upper, t in _HK_TICKS if price < upper)
    elif mkt == "US":
        tick = 0.0001 if price < 1 else 0.01
    else:
        tick = 0.001 if code.startswith(("5", "15", "16", "18")) else 0.01
    return round(round(price / tick) * tick, 6)


# =========================================================================
# shared helpers
# =========================================================================
def _import_first(names: Sequence[str], hint: str) -> ModuleType:
    """Import the first available module of ``names`` or raise GatewayUnavailable."""
    errors = []
    for name in names:
        try:
            return importlib.import_module(name)
        except ImportError as e:  # ModuleNotFoundError included
            errors.append(f"{name}: {e}")
    raise GatewayUnavailable(f"{hint}\n(import 失败: {'; '.join(errors)})")


def _fx(registry: MarketRegistry, currency: str) -> Optional[float]:
    rates = registry.fx_rates
    if currency in rates:
        return rates[currency]
    alias = {"CNH": "CNY", "CNY": "CNH"}.get(currency)
    return rates.get(alias) if alias else None


def combine_balances(parts: Iterable[tuple[float, float, str]],
                     registry: Optional[MarketRegistry] = None) -> AccountSnapshot:
    """Merge ``(cash, total_value, currency)`` balances into one snapshot.

    With a ``registry`` every part is converted into ``registry.base_currency``
    when its FX rate is known; otherwise all parts must share one currency
    (which is then reported in ``AccountSnapshot.currency``).
    """
    parts = [(float(c), float(t), (cur or "").upper()) for c, t, cur in parts]
    if not parts:
        raise GatewayError("未查询到账户资金")
    if registry is not None:
        rates = [_fx(registry, cur) for _, _, cur in parts]
        if all(r is not None for r in rates):
            return AccountSnapshot(cash=sum(c * r for (c, _, _), r in zip(parts, rates)),
                                   total_value=sum(t * r for (_, t, _), r in zip(parts, rates)),
                                   currency=registry.base_currency)
    currencies = {cur for _, _, cur in parts}
    if len(currencies) > 1:
        raise GatewayError(f"账户含多个币种 {sorted(currencies)}，请提供 registry 及汇率 fx_rates")
    cur = currencies.pop()
    if registry is not None and cur != registry.base_currency:
        log.warning("缺少 %s->%s 汇率，账户金额以 %s 报告", cur, registry.base_currency, cur)
    return AccountSnapshot(cash=sum(p[0] for p in parts), total_value=sum(p[1] for p in parts),
                           currency=cur)


def _qty(x: Any) -> int:
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return 0


def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _pick(d: dict, keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _position(symbol: str, qty: int, sellable: Optional[int], cost: float, last: float) -> Position:
    sellable = qty if sellable is None else sellable
    return Position(symbol, quantity=qty, avg_cost=cost, last_price=last,
                    today_bought=max(qty - max(sellable, 0), 0), peak_price=last)


# =========================================================================
# Interactive Brokers
# =========================================================================
class IBGateway(BaseGateway):
    """IBKR via TWS / IB Gateway using ``ib_async`` (maintained fork) or ``ib_insync``.

    Ports: TWS paper 7497 / live 7496; IB Gateway paper 4002 / live 4001.
    Enable *Configure → API → Settings → Enable ActiveX and Socket Clients*
    (and untick *Read-Only API* to trade). A-shares go through Stock Connect
    northbound (``SEHKNTL`` / ``SEHKSZSE``, CNH, limit orders only) and need
    the China Connect trading permission; set ``china_connect=False`` to block.
    In Jupyter call ``ib_insync.util.startLoop()`` first.
    """
    name = "ibkr"
    markets = ("US", "HK", "CN")
    PAPER_PORTS = (7497, 4002)
    _HINT = ("需要 IBKR Python SDK: pip install ib_async  (或 pip install ib_insync)，"
             "并启动 TWS 或 IB Gateway 且开启 API 端口")

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1,
                 account: str = "", readonly: bool = False, timeout: float = 10.0,
                 tif: str = "DAY", china_connect: bool = True, paper: Optional[bool] = None,
                 registry: Optional[MarketRegistry] = None, order_wait: float = 0.5,
                 cash_tag: Optional[str] = None):
        self.host, self.port, self.client_id = host, int(port), int(client_id)
        self.account, self.readonly, self.timeout, self.tif = account, readonly, timeout, tif
        self.china_connect, self.registry, self.order_wait = china_connect, registry, order_wait
        self.cash_tag = cash_tag          # None = min(TotalCashValue, AvailableFunds)
        self._paper_opt = paper
        self.paper = bool(paper) if paper is not None else self.port in self.PAPER_PORTS
        self.base_currency = ""
        self.ib: Any = None
        self._mod: Any = None
        self._contracts: dict[str, Any] = {}

    # ---------------------------------------------------------------- session
    def connect(self) -> None:
        self._mod = _import_first(("ib_async", "ib_insync"), self._HINT)
        self.ib = self._mod.IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout,
                            readonly=self.readonly, account=self.account)
        except Exception as e:  # ConnectionRefusedError, TimeoutError, ...
            raise GatewayUnavailable(
                f"无法连接 TWS/IB Gateway {self.host}:{self.port} (clientId={self.client_id}): {e!r}。"
                "请确认已登录、API 已启用、端口正确 (TWS 7497 模拟/7496 实盘, Gateway 4002/4001)，"
                "且 clientId 未被占用") from e
        accounts = list(self.ib.managedAccounts())
        if not self.account:
            if not accounts:
                raise GatewayError("IBKR 未返回任何账户")
            self.account = accounts[0]
            if len(accounts) > 1:
                log.warning("IBKR 有多个账户 %s，默认使用 %s (可用 account= 指定)", accounts, self.account)
        elif accounts and self.account not in accounts:
            raise GatewayError(f"账户 {self.account} 不在 {accounts} 中")
        self.paper = self._infer_paper(self.port, self.account, self._paper_opt)
        log.info("IBKR connected account=%s paper=%s", self.account, self.paper)

    @classmethod
    def _infer_paper(cls, port: int, account: str = "", explicit: Optional[bool] = None) -> bool:
        """Paper accounts start with ``D`` (``DU1234567``); else fall back to the port."""
        if explicit is not None:
            return bool(explicit)
        if account:
            return account.upper().startswith("D")
        return int(port) in cls.PAPER_PORTS

    def close(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()

    # ---------------------------------------------------------------- account
    def query_account(self) -> AccountSnapshot:
        vals: dict[str, tuple[float, str]] = {}
        for r in self.ib.accountSummary(self.account):
            if r.account == self.account and r.tag in ("NetLiquidation", "TotalCashValue",
                                                       "AvailableFunds"):
                vals[r.tag] = (float(r.value), r.currency)
        if "NetLiquidation" not in vals:     # fallback: account updates stream
            for r in self.ib.accountValues(self.account):
                if r.tag in ("NetLiquidation", "TotalCashValue", "AvailableFunds") \
                        and r.currency not in ("", "BASE") and r.tag not in vals:
                    vals[r.tag] = (float(r.value), r.currency)
        if "NetLiquidation" not in vals:
            raise GatewayError("IBKR accountSummary 未返回 NetLiquidation")
        total, cur = vals["NetLiquidation"]
        self.base_currency = cur
        if self.cash_tag:
            if self.cash_tag not in vals:
                raise GatewayError(f"IBKR 未返回 {self.cash_tag}")
            cash = vals[self.cash_tag][0]
        else:
            cash = min(v[0] for k, v in vals.items() if k in ("TotalCashValue", "AvailableFunds")) \
                if any(k in vals for k in ("TotalCashValue", "AvailableFunds")) else 0.0
        return combine_balances([(cash, total, cur)], self.registry)

    def query_positions(self) -> dict[str, Position]:
        prices = {}
        try:
            for it in self.ib.portfolio():
                if it.account == self.account:
                    prices[it.contract.conId] = float(it.marketPrice)
        except Exception:  # portfolio needs the account updates subscription
            pass
        out: dict[str, Position] = {}
        for p in self.ib.positions(self.account):
            c = p.contract
            if c.secType != "STK" or not p.position:
                continue
            sym = from_ib(c.symbol, c.currency, c.exchange or c.primaryExchange)
            # IB does not report sellable qty; A-share (Stock Connect) T+1 is not tracked here
            out[sym] = _position(sym, _qty(p.position), None, float(p.avgCost),
                                 _num(prices.get(c.conId)))
        return out

    # ---------------------------------------------------------------- orders
    def contract(self, symbol: str) -> Any:
        """Qualified ``Stock`` contract for ``symbol`` (cached)."""
        if symbol not in self._contracts:
            sym, exchange, currency = to_ib(symbol)
            if exchange.startswith("SEHK") and exchange != "SEHK" and not self.china_connect:
                raise GatewayError(f"未启用沪深港通 (china_connect=False): {symbol}")
            c = self._mod.Stock(sym, exchange, currency)
            q = [x for x in (self.ib.qualifyContracts(c) or []) if x is not None]
            if not q or not getattr(q[0], "conId", 0):
                raise GatewayError(f"IBKR 无法识别合约 {symbol} -> Stock({sym!r},{exchange!r},{currency!r})")
            self._contracts[symbol] = q[0]
        return self._contracts[symbol]

    def send_order(self, order: Order) -> str:
        contract = self.contract(order.symbol)
        action = "BUY" if order.direction == Direction.BUY else "SELL"
        kw = dict(tif=self.tif, account=self.account, orderRef=(order.tag or order.order_id)[:40])
        if order.price is None:
            if contract.currency == "CNH":
                raise GatewayError("沪深港通(北向)不支持市价单，请为 A 股订单指定 order.price")
            ib_order = self._mod.MarketOrder(action, order.quantity, **kw)
        else:
            ib_order = self._mod.LimitOrder(action, order.quantity,
                                            round_price(order.symbol, order.price), **kw)
        trade = self.ib.placeOrder(contract, ib_order)
        if self.order_wait > 0:
            self.ib.sleep(self.order_wait)
        status = trade.orderStatus.status
        if status in ("Cancelled", "ApiCancelled", "Inactive"):
            msg = "; ".join(e.message for e in trade.log if e.message) or status
            raise GatewayError(f"IBKR 拒单 {order.symbol}: {msg}")
        return str(trade.order.orderId)

    def _find_trade(self, broker_order_id: str) -> Any:
        for t in list(self.ib.openTrades()) + list(self.ib.trades()):
            if broker_order_id in (str(t.order.orderId), str(t.order.permId)):
                return t
        return None

    def cancel_order(self, broker_order_id: str) -> None:
        t = self._find_trade(broker_order_id)
        if t is None:
            raise GatewayError(f"IBKR 未找到订单 {broker_order_id} (仅能撤本 clientId 的订单)")
        self.ib.cancelOrder(t.order)

    def query_open_orders(self) -> list[dict]:
        out = []
        for t in self.ib.openTrades():
            c, o = t.contract, t.order
            out.append(dict(order_id=str(o.orderId),
                            symbol=from_ib(c.symbol, c.currency, c.exchange or c.primaryExchange),
                            direction="buy" if o.action == "BUY" else "sell",
                            quantity=_qty(o.totalQuantity),
                            price=o.lmtPrice if o.orderType == "LMT" else None,
                            filled=_qty(t.orderStatus.filled), status=t.orderStatus.status))
        return out


# =========================================================================
# Futu / moomoo
# =========================================================================
class FutuGateway(BaseGateway):
    """Futu securities via OpenD (``pip install futu-api``).

    One ``OpenSecTradeContext`` per market (``markets=("HK", "US")``). For REAL
    trading pass ``trade_password`` (or unlock in the OpenD GUI). A-shares via
    Stock Connect are not supported here. ``security_firm`` is one of
    FUTUSECURITIES (香港) / FUTUINC (moomoo US) / FUTUSG / FUTUAU.
    """
    name = "futu"
    markets = ("HK", "US")
    _HINT = "需要富途 OpenAPI: pip install futu-api ，并启动 FutuOpenD (默认 127.0.0.1:11111) 且登录"

    def __init__(self, host: str = "127.0.0.1", port: int = 11111, paper: bool = True,
                 markets: Sequence[str] = ("HK", "US"), trade_password: Optional[str] = None,
                 security_firm: str = "FUTUSECURITIES", acc_id: int = 0,
                 currency: Optional[str] = None, registry: Optional[MarketRegistry] = None):
        self.host, self.port, self.paper = host, int(port), paper
        self.markets = tuple(m.upper() for m in markets)
        self.trade_password = trade_password or os.environ.get("FUTU_TRADE_PASSWORD")
        self.security_firm = security_firm
        self.acc_id, self.currency, self.registry = int(acc_id), currency, registry
        self.ft: Any = None
        self._ctx: dict[str, Any] = {}
        self._acc: dict[str, int] = {}
        self._order_mkt: dict[str, str] = {}

    @property
    def _env(self) -> Any:
        return self.ft.TrdEnv.SIMULATE if self.paper else self.ft.TrdEnv.REAL

    def _call(self, what: str, fn, *a, **kw) -> Any:
        ret, data = fn(*a, **kw)
        if ret != self.ft.RET_OK:
            raise GatewayError(f"富途 {what} 失败: {data}")
        return data

    def connect(self) -> None:
        self.ft = _import_first(("futu",), self._HINT)
        ft = self.ft
        firm = getattr(ft.SecurityFirm, self.security_firm)
        env_name = "SIMULATE" if self.paper else "REAL"
        for m in self.markets:
            ctx = ft.OpenSecTradeContext(filter_trdmarket=getattr(ft.TrdMarket, m),
                                         host=self.host, port=self.port, security_firm=firm)
            self._ctx[m] = ctx
            ret, accs = ctx.get_acc_list()
            if ret != ft.RET_OK:
                self.close()
                raise GatewayUnavailable(f"无法连接 FutuOpenD {self.host}:{self.port}: {accs}")
            accs = accs[accs["trd_env"].astype(str) == env_name]
            if "sim_acc_type" in accs and self.paper:
                stock = accs[accs["sim_acc_type"].astype(str) == "STOCK"]
                accs = stock if len(stock) else accs
            if self.acc_id:
                accs = accs[accs["acc_id"] == self.acc_id]
            if accs.empty:
                self.close()
                raise GatewayError(f"富途未找到 {m} 市场 {env_name} 账户")
            self._acc[m] = int(accs["acc_id"].iloc[0])
            if not self.paper and self.trade_password:
                self._call("解锁交易", ctx.unlock_trade, password=self.trade_password)

    def close(self) -> None:
        for ctx in self._ctx.values():
            try:
                ctx.close()
            except Exception:
                pass
        self._ctx.clear()

    def _market(self, symbol: str) -> str:
        m = split_symbol(symbol)[1]
        if m not in self._ctx:
            raise GatewayError(f"富途网关未开启 {m} 市场: {symbol}")
        return m

    def query_account(self) -> AccountSnapshot:
        parts, seen = [], set()
        for m, ctx in self._ctx.items():
            acc = self._acc[m]
            if acc in seen:          # universal account shared by HK and US contexts
                continue
            seen.add(acc)
            kw = dict(trd_env=self._env, acc_id=acc, refresh_cache=True)
            if self.currency:
                kw["currency"] = getattr(self.ft.Currency, self.currency)
            df = self._call("查询资金", ctx.accinfo_query, **kw)
            row = df.iloc[0]
            cur = str(row["currency"]) if "currency" in df and row["currency"] not in (None, "N/A") \
                else (self.currency or {"HK": "HKD", "US": "USD"}[m])
            parts.append((_num(row["cash"]), _num(row["total_assets"]), cur))
        return combine_balances(parts, self.registry)

    def query_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for m, ctx in self._ctx.items():
            df = self._call("查询持仓", ctx.position_list_query, trd_env=self._env,
                            acc_id=self._acc[m], refresh_cache=True)
            for _, r in df.iterrows():
                if not str(r["code"]).upper().startswith(m + "."):
                    continue
                qty = _qty(r["qty"])
                if qty == 0 or str(r.get("position_side", "LONG")) == "SHORT":
                    continue
                sym = from_futu(str(r["code"]))
                cost = _num(r.get("average_cost", r.get("cost_price")))
                if cost == 0:
                    cost = _num(r.get("cost_price"))
                out[sym] = _position(sym, qty, _qty(r.get("can_sell_qty", qty)), cost,
                                     _num(r.get("nominal_price")))
        return out

    def send_order(self, order: Order) -> str:
        m = self._market(order.symbol)
        ft = self.ft
        is_mkt = order.price is None
        df = self._call(
            "下单", self._ctx[m].place_order,
            price=0.0 if is_mkt else round_price(order.symbol, order.price),
            qty=order.quantity, code=to_futu(order.symbol),
            trd_side=ft.TrdSide.BUY if order.direction == Direction.BUY else ft.TrdSide.SELL,
            order_type=ft.OrderType.MARKET if is_mkt else ft.OrderType.NORMAL,
            trd_env=self._env, acc_id=self._acc[m], remark=(order.tag or order.order_id)[:64],
        )
        oid = str(df["order_id"].iloc[0])
        self._order_mkt[oid] = m
        return oid

    def cancel_order(self, broker_order_id: str) -> None:
        markets = [self._order_mkt[broker_order_id]] if broker_order_id in self._order_mkt \
            else list(self._ctx)
        err = None
        for m in markets:
            ret, data = self._ctx[m].modify_order(self.ft.ModifyOrderOp.CANCEL, broker_order_id,
                                                  0, 0, trd_env=self._env, acc_id=self._acc[m])
            if ret == self.ft.RET_OK:
                return
            err = data
        raise GatewayError(f"富途撤单失败 {broker_order_id}: {err}")

    def query_open_orders(self) -> list[dict]:
        active = {"WAITING_SUBMIT", "SUBMITTING", "SUBMITTED", "FILLED_PART"}
        out = []
        for m, ctx in self._ctx.items():
            df = self._call("查询订单", ctx.order_list_query, trd_env=self._env,
                            acc_id=self._acc[m], refresh_cache=True)
            for _, r in df.iterrows():
                if str(r["order_status"]) in active:
                    out.append(dict(order_id=str(r["order_id"]), symbol=from_futu(str(r["code"])),
                                    direction="buy" if str(r["trd_side"]) == "BUY" else "sell",
                                    quantity=_qty(r["qty"]), price=_num(r["price"]),
                                    filled=_qty(r.get("dealt_qty", 0)), status=str(r["order_status"])))
        return out


# =========================================================================
# Tiger Brokers
# =========================================================================
class TigerGateway(BaseGateway):
    """Tiger Open API (``pip install tigeropen``).

    ``props_path`` is the directory holding ``tiger_openapi_config.properties``
    (downloaded from the developer page). ``account`` overrides the account in
    that file - use your paper account id for 模拟交易 and set ``paper=True``.
    """
    name = "tiger"
    markets = ("HK", "US")
    _HINT = "需要老虎证券 OpenAPI: pip install tigeropen ，并在开发者页面下载 tiger_openapi_config.properties"

    def __init__(self, props_path: Optional[str] = None, account: Optional[str] = None,
                 paper: bool = False, registry: Optional[MarketRegistry] = None):
        self.props_path, self.account, self.paper, self.registry = props_path, account, paper, registry
        self.client: Any = None
        self._m: dict[str, Any] = {}

    def connect(self) -> None:
        _import_first(("tigeropen",), self._HINT)
        try:
            from tigeropen.common.consts import Currency, Market, SecurityType
            from tigeropen.common.util.contract_utils import stock_contract
            from tigeropen.common.util.order_utils import limit_order, market_order
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.trade.trade_client import TradeClient
        except ImportError as e:
            raise GatewayUnavailable(f"{self._HINT} ({e})") from e
        self._m = dict(stock_contract=stock_contract, market_order=market_order,
                       limit_order=limit_order, Currency=Currency, Market=Market,
                       SecurityType=SecurityType)
        try:
            config = TigerOpenClientConfig(props_path=self.props_path) if self.props_path \
                else TigerOpenClientConfig()
            if self.account:
                config.account = self.account
            self.account = config.account
            self.client = TradeClient(config)
            self.client.get_managed_accounts()
        except Exception as e:
            raise GatewayUnavailable(f"老虎 OpenAPI 初始化失败: {e!r} (检查 props_path / tiger_id / 私钥)") from e

    def close(self) -> None:
        self.client = None

    def query_account(self) -> AccountSnapshot:
        try:  # prime / 综合账户 and paper accounts
            assets = self.client.get_prime_assets(account=self.account)
            seg = assets.segments.get("S")
            cash = _attr(seg, ("cash_available_for_trade", "cash_balance"), 0.0)
            return combine_balances([(_num(cash), _num(seg.net_liquidation),
                                      _attr(seg, ("currency",), "USD"))], self.registry)
        except Exception as e:  # legacy global accounts
            log.debug("get_prime_assets failed (%s), falling back to get_assets", e)
        pa = self.client.get_assets(account=self.account)[0]
        s = pa.summary
        cash = _attr(s, ("available_funds", "cash"), 0.0)
        return combine_balances([(_num(cash), _num(s.net_liquidation),
                                  _attr(s, ("currency",), "USD"))], self.registry)

    def query_positions(self) -> dict[str, Position]:
        m = self._m
        out: dict[str, Position] = {}
        for p in self.client.get_positions(account=self.account, sec_type=m["SecurityType"].STK,
                                           currency=m["Currency"].ALL, market=m["Market"].ALL) or []:
            c = p.contract
            qty = _qty(_attr(p, ("position_qty", "quantity", "position"), 0))
            if qty <= 0:
                continue
            sym = from_tiger(c.symbol, str(_attr(c, ("currency",), "")))
            sellable = _attr(p, ("salable_qty", "salable"))
            out[sym] = _position(sym, qty, None if sellable is None else _qty(sellable),
                                 _num(p.average_cost), _num(p.market_price))
        return out

    def send_order(self, order: Order) -> str:
        m = self._m
        sym, cur = to_tiger(order.symbol)
        contract = m["stock_contract"](symbol=sym, currency=cur)
        action = "BUY" if order.direction == Direction.BUY else "SELL"
        if order.price is None:
            o = m["market_order"](account=self.account, contract=contract, action=action,
                                  quantity=order.quantity)
        else:
            o = m["limit_order"](account=self.account, contract=contract, action=action,
                                 quantity=order.quantity,
                                 limit_price=round_price(order.symbol, order.price))
        ret = self.client.place_order(o)
        oid = getattr(o, "id", None) or ret
        if not oid:
            raise GatewayError(f"老虎下单未返回订单号: {ret}")
        return str(oid)

    def cancel_order(self, broker_order_id: str) -> None:
        self.client.cancel_order(account=self.account, id=int(broker_order_id))

    def query_open_orders(self) -> list[dict]:
        out = []
        for o in self.client.get_open_orders(account=self.account) or []:
            c = o.contract
            out.append(dict(order_id=str(o.id), symbol=from_tiger(c.symbol, str(_attr(c, ("currency",), ""))),
                            direction="buy" if str(o.action).upper() == "BUY" else "sell",
                            quantity=_qty(o.quantity), price=getattr(o, "limit_price", None),
                            filled=_qty(getattr(o, "filled", 0)), status=str(o.status)))
        return out


# =========================================================================
# Longbridge / LongPort
# =========================================================================
class LongbridgeGateway(BaseGateway):
    """LongPort OpenAPI (``pip install longport``; formerly ``longbridge``).

    Credentials come from env vars ``LONGPORT_APP_KEY`` / ``LONGPORT_APP_SECRET``
    / ``LONGPORT_ACCESS_TOKEN`` (old SDK: ``LONGBRIDGE_*``) or the constructor.
    Use a paper-account access token for 模拟交易 and set ``paper=True``.
    """
    name = "longbridge"
    markets = ("HK", "US", "CN")
    _HINT = ("需要长桥 OpenAPI: pip install longport ，并设置环境变量 LONGPORT_APP_KEY / "
             "LONGPORT_APP_SECRET / LONGPORT_ACCESS_TOKEN (开发者中心获取)")

    def __init__(self, app_key: Optional[str] = None, app_secret: Optional[str] = None,
                 access_token: Optional[str] = None, paper: bool = False,
                 currency: Optional[str] = None, registry: Optional[MarketRegistry] = None):
        self.app_key, self.app_secret, self.access_token = app_key, app_secret, access_token
        self.paper, self.currency, self.registry = paper, currency, registry
        self.lp: Any = None
        self.ctx: Any = None

    def connect(self) -> None:
        self.lp = _import_first(("longport.openapi", "longbridge.openapi"), self._HINT)
        try:
            if self.app_key:
                config = self.lp.Config(app_key=self.app_key, app_secret=self.app_secret,
                                        access_token=self.access_token)
            else:
                config = self.lp.Config.from_env()
            self.ctx = self.lp.TradeContext(config)
        except Exception as e:
            raise GatewayUnavailable(f"长桥 TradeContext 创建失败: {e!r}\n{self._HINT}") from e

    def close(self) -> None:
        self.ctx = None

    def query_account(self) -> AccountSnapshot:
        balances = self.ctx.account_balance(self.currency) if self.currency \
            else self.ctx.account_balance()
        if not balances:
            raise GatewayError("长桥 account_balance 返回为空")
        b = balances[0]
        return combine_balances([(_num(b.total_cash), _num(b.net_assets), str(b.currency))],
                                self.registry)

    def query_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for ch in self.ctx.stock_positions().channels:
            for p in ch.positions:
                qty = _qty(p.quantity)
                if qty <= 0:
                    continue
                sym = from_longport(p.symbol)
                out[sym] = _position(sym, qty, _qty(p.available_quantity), _num(p.cost_price), 0.0)
        return out

    def send_order(self, order: Order) -> str:
        lp = self.lp
        kw: dict[str, Any] = dict(
            symbol=to_longport(order.symbol),
            order_type=lp.OrderType.MO if order.price is None else lp.OrderType.LO,
            side=lp.OrderSide.Buy if order.direction == Direction.BUY else lp.OrderSide.Sell,
            submitted_quantity=Decimal(order.quantity),
            time_in_force=lp.TimeInForceType.Day,
            remark=(order.tag or order.order_id)[:64],
        )
        if order.price is not None:
            kw["submitted_price"] = Decimal(str(round_price(order.symbol, order.price)))
        try:
            resp = self.ctx.submit_order(**kw)
        except TypeError:  # older SDKs take an int quantity
            kw["submitted_quantity"] = int(order.quantity)
            resp = self.ctx.submit_order(**kw)
        return str(resp.order_id)

    def cancel_order(self, broker_order_id: str) -> None:
        self.ctx.cancel_order(broker_order_id)

    def query_open_orders(self) -> list[dict]:
        active = ("NotReported", "New", "WaitToNew", "PartialFilled", "PendingReplace", "WaitToReplace")
        out = []
        for o in self.ctx.today_orders():
            st = str(o.status).rsplit(".", 1)[-1]
            if st in active:
                out.append(dict(order_id=str(o.order_id), symbol=from_longport(o.symbol),
                                direction="buy" if "Buy" in str(o.side) else "sell",
                                quantity=_qty(o.quantity), price=_num(o.price) or None,
                                filled=_qty(o.executed_quantity), status=st))
        return out


# =========================================================================
# miniQMT (xtquant)
# =========================================================================
class QMTGateway(BaseGateway):
    """A-shares through miniQMT (``xtquant``; Windows, QMT client in 极简模式 logged in).

    ``path`` is the client's ``userdata_mini`` directory, e.g.
    ``D:\\国金QMT交易端模拟\\userdata_mini``. Market orders use 最优五档即时成交
    剩余撤销; for BJ (and when ``use_market=False``) a limit order at
    ``reference * (1 ± protect_pct)`` is sent, with the reference taken from
    ``xtdata.get_full_tick``. Pass ``order.price`` to control the price yourself.
    """
    name = "qmt"
    markets = ("CN",)
    _HINT = ("需要 miniQMT 的 xtquant: pip install xtquant (或从券商 QMT 安装目录复制 xtquant 包)，"
             "并以极简模式登录 QMT 客户端")

    def __init__(self, path: str = "", account_id: str = "", session_id: Optional[int] = None,
                 paper: bool = False, use_market: bool = True, protect_pct: float = 0.02,
                 strategy_name: str = "quantum_analyze"):
        self.path, self.account_id, self.paper = path, str(account_id), paper
        self.session_id = session_id or int(time.time()) % 1_000_000
        self.use_market, self.protect_pct, self.strategy_name = use_market, protect_pct, strategy_name
        self.trader: Any = None
        self.acc: Any = None
        self.xc: Any = None
        self.xtdata: Any = None

    def connect(self) -> None:
        _import_first(("xtquant",), self._HINT)
        try:
            from xtquant import xtconstant
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError as e:
            raise GatewayUnavailable(f"{self._HINT} ({e})") from e
        try:
            from xtquant import xtdata
            self.xtdata = xtdata
        except ImportError:
            self.xtdata = None
        if not self.path or not self.account_id:
            raise GatewayError("QMT 需要 path (userdata_mini 目录) 和 account_id")
        self.xc = xtconstant
        self.trader = XtQuantTrader(self.path, self.session_id)
        self.trader.start()
        r = self.trader.connect()
        if r != 0:
            raise GatewayUnavailable(f"连接 miniQMT 失败 (返回 {r})：请确认 QMT 已以极简模式登录，path={self.path}")
        self.acc = StockAccount(self.account_id)
        r = self.trader.subscribe(self.acc)
        if r != 0:
            log.warning("QMT subscribe(%s) 返回 %s", self.account_id, r)

    def close(self) -> None:
        if self.trader is not None:
            try:
                self.trader.stop()
            finally:
                self.trader = None

    def query_account(self) -> AccountSnapshot:
        a = self.trader.query_stock_asset(self.acc)
        if a is None:
            raise GatewayError("QMT query_stock_asset 返回 None (未连接或账户未订阅)")
        return AccountSnapshot(cash=float(a.cash), total_value=float(a.total_asset), currency="CNY")

    def query_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self.trader.query_stock_positions(self.acc) or []:
            qty = _qty(p.volume)
            if qty <= 0:
                continue
            sym = from_qmt(p.stock_code)
            last = _num(p.market_value) / qty if qty else 0.0
            out[sym] = _position(sym, qty, _qty(p.can_use_volume),
                                 _num(_attr(p, ("avg_price", "open_price"), 0.0)), last)
        return out

    def _reference_price(self, code: str) -> Optional[float]:
        if self.xtdata is None:
            return None
        try:
            tick = self.xtdata.get_full_tick([code]).get(code) or {}
            px = _num(tick.get("lastPrice")) or _num(tick.get("lastClose"))
            return px or None
        except Exception:
            return None

    def protective_price(self, symbol: str, direction: Direction, ref: float) -> float:
        pct = self.protect_pct if direction == Direction.BUY else -self.protect_pct
        return round_price(symbol, ref * (1 + pct))

    def send_order(self, order: Order) -> str:
        xc = self.xc
        code = to_qmt(order.symbol)
        side = xc.STOCK_BUY if order.direction == Direction.BUY else xc.STOCK_SELL
        mkt = code.rsplit(".", 1)[1]
        if order.price is not None:
            price_type, price = xc.FIX_PRICE, round_price(order.symbol, order.price)
        else:
            ref = self._reference_price(code)
            if self.use_market and mkt in ("SH", "SZ"):
                price_type = xc.MARKET_SH_CONVERT_5_CANCEL if mkt == "SH" else xc.MARKET_SZ_CONVERT_5_CANCEL
                # protective price is required for STAR (688) market orders; harmless otherwise
                price = self.protective_price(order.symbol, order.direction, ref) if ref else 0.0
            elif ref:
                price_type, price = xc.FIX_PRICE, self.protective_price(order.symbol, order.direction, ref)
            else:
                raise GatewayError(f"{order.symbol} 无法获取参考价，请指定 order.price")
        oid = self.trader.order_stock(self.acc, code, side, int(order.quantity), price_type,
                                      float(price), self.strategy_name,
                                      (order.tag or order.order_id)[:20])
        if oid is None or oid < 0:
            raise GatewayError(f"QMT 下单失败 {order.symbol} (返回 {oid})")
        return str(oid)

    def cancel_order(self, broker_order_id: str) -> None:
        r = self.trader.cancel_order_stock(self.acc, int(broker_order_id))
        if r != 0:
            raise GatewayError(f"QMT 撤单失败 {broker_order_id} (返回 {r})")

    def query_open_orders(self) -> list[dict]:
        out = []
        for o in self.trader.query_stock_orders(self.acc, cancelable_only=True) or []:
            out.append(dict(order_id=str(o.order_id), symbol=from_qmt(o.stock_code),
                            direction="buy" if o.order_type == self.xc.STOCK_BUY else "sell",
                            quantity=_qty(o.order_volume), price=_num(o.price),
                            filled=_qty(o.traded_volume), status=str(o.order_status)))
        return out


# =========================================================================
# easytrader (least reliable: GUI automation)
# =========================================================================
class EasyTraderGateway(BaseGateway):
    """A-shares by driving the 同花顺 / 通用版 desktop client (``pip install easytrader``).

    **Least reliable fallback**: it clicks through a Windows GUI, breaks when the
    client updates or a popup appears, and cannot run while the screen is locked.
    ``client`` is an ``easytrader.use`` name (``universal_client``, ``ths``,
    ``ht_client``, ``gj_client``...); ``exe_path`` is e.g. ``C:\\同花顺软件\\同花顺\\xiadan.exe``.
    """
    name = "easytrader"
    markets = ("CN",)
    _HINT = "需要 easytrader (仅 Windows): pip install easytrader ，并登录同花顺/通用版下单客户端"

    _CASH = ("可用金额", "可用资金", "资金余额")
    _TOTAL = ("总资产", "资产总值", "总 资 产")
    _CODE = ("证券代码", "股票代码")
    _QTY = ("股票余额", "当前持仓", "实际数量", "证券数量", "持股数量", "股份余额")
    _AVAIL = ("可用余额", "可卖数量", "可用数量", "可用股份")
    _COST = ("成本价", "参考成本价", "买入成本", "摊薄成本价")
    _PRICE = ("市价", "当前价", "最新价", "参考市价")

    def __init__(self, client: str = "universal_client", exe_path: Optional[str] = None,
                 config_path: Optional[str] = None, paper: bool = False):
        self.client, self.exe_path, self.config_path, self.paper = client, exe_path, config_path, paper
        self.user: Any = None

    def connect(self) -> None:
        et = _import_first(("easytrader",), self._HINT)
        if not sys.platform.startswith("win"):
            raise GatewayUnavailable("easytrader 依赖 Windows 桌面客户端，当前系统不支持")
        try:
            self.user = et.use(self.client)
            if self.exe_path:
                self.user.connect(self.exe_path)
            elif self.config_path:
                self.user.prepare(self.config_path)
            else:
                raise GatewayError("easytrader 需要 exe_path (connect) 或 config_path (prepare)")
        except GatewayError:
            raise
        except Exception as e:
            raise GatewayUnavailable(f"easytrader 连接客户端失败: {e!r}") from e

    def close(self) -> None:
        self.user = None

    def query_account(self) -> AccountSnapshot:
        b = self.user.balance
        b = b[0] if isinstance(b, list) else b
        return AccountSnapshot(cash=_num(_pick(b, self._CASH)), total_value=_num(_pick(b, self._TOTAL)),
                               currency="CNY")

    def query_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for r in self.user.position or []:
            qty = _qty(_pick(r, self._QTY, 0))
            if qty <= 0:
                continue
            sym = from_easytrader(_pick(r, self._CODE))
            out[sym] = _position(sym, qty, _qty(_pick(r, self._AVAIL, qty)),
                                 _num(_pick(r, self._COST)), _num(_pick(r, self._PRICE)))
        return out

    def send_order(self, order: Order) -> str:
        code = to_easytrader(order.symbol)
        buy = order.direction == Direction.BUY
        if order.price is None:
            fn = self.user.market_buy if buy else self.user.market_sell
            r = fn(code, amount=order.quantity)
        else:
            fn = self.user.buy if buy else self.user.sell
            r = fn(code, price=round_price(order.symbol, order.price), amount=order.quantity)
        oid = r.get("entrust_no") if isinstance(r, dict) else None
        if not oid:
            raise GatewayError(f"easytrader 下单未返回委托号: {r}")
        return str(oid)

    def cancel_order(self, broker_order_id: str) -> None:
        self.user.cancel_entrust(broker_order_id)


# =========================================================================
# Manual / CSV and Ptrade export
# =========================================================================
_ORDER_FIELDS = ["time", "order_id", "action", "symbol", "code", "direction", "quantity",
                 "price", "tag", "status"]


class ManualGateway(BaseGateway):
    """Writes orders to a CSV for manual entry (or another program) and reads
    the account from a user-maintained file.

    ``state_path`` may be JSON::

        {"cash": 50000, "currency": "CNY",
         "positions": [{"symbol": "600519.SH", "quantity": 100, "avg_cost": 1500,
                        "last_price": 1520, "today_bought": 0}]}

    or CSV with columns ``symbol,quantity,avg_cost,last_price[,today_bought]``
    plus an optional row with symbol ``CASH`` whose ``quantity`` is the cash.
    ``total_value`` = cash + market value (converted with ``registry`` if given)
    unless the JSON provides ``total_value``. ``code_format`` adds a broker code
    column (standard / ptrade / qmt / easytrader / futu / longport / ib).
    """
    name = "manual"
    markets = ("CN", "HK", "US")

    def __init__(self, orders_path: str = "orders_manual.csv", state_path: Optional[str] = None,
                 cash: float = 0.0, currency: str = "CNY", code_format: str = "standard",
                 registry: Optional[MarketRegistry] = None, paper: bool = False):
        if code_format not in _CODE_FORMATS:
            raise GatewayError(f"未知 code_format {code_format}，可选 {sorted(_CODE_FORMATS)}")
        self.orders_path = Path(orders_path)
        self.state_path = Path(state_path) if state_path else None
        self.cash, self.currency, self.code_format = float(cash), currency, code_format
        self.registry, self.paper = registry, paper
        self.open_orders: dict[str, Order] = {}
        self._seq = 0

    def connect(self) -> None:
        self.orders_path.parent.mkdir(parents=True, exist_ok=True)
        if self.orders_path.exists():
            with self.orders_path.open(newline="", encoding="utf-8-sig") as f:
                self._seq = sum(1 for r in csv.DictReader(f) if r.get("action") == "new")

    def close(self) -> None:
        pass

    # ---------------------------------------------------------------- state file
    def _load_state(self) -> tuple[float, Optional[float], str, dict[str, Position]]:
        cash, total, cur, positions = self.cash, None, self.currency, {}
        p = self.state_path
        if p is None or not p.exists():
            return cash, total, cur, positions
        if p.suffix.lower() == ".json":
            d = json.loads(p.read_text(encoding="utf-8"))
            cash = float(d.get("cash", cash))
            total = d.get("total_value")
            cur = d.get("currency", cur)
            rows = d.get("positions", [])
            if isinstance(rows, dict):
                rows = [{"symbol": k, **v} for k, v in rows.items()]
        else:
            with p.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            for r in [r for r in rows if str(r.get("symbol", "")).upper() == "CASH"]:
                cash = _num(r.get("quantity"))
                rows.remove(r)
        for r in rows:
            sym = "{}.{}".format(*split_symbol(str(r["symbol"])))
            qty = _qty(r.get("quantity"))
            if qty > 0:
                tb = _qty(r.get("today_bought") or 0)
                positions[sym] = _position(sym, qty, qty - tb, _num(r.get("avg_cost")),
                                           _num(r.get("last_price")))
        return cash, (None if total is None else float(total)), cur, positions

    def query_account(self) -> AccountSnapshot:
        cash, total, cur, positions = self._load_state()
        if total is None:
            mv = 0.0
            for s, p in positions.items():
                fx = 1.0
                if self.registry is not None:
                    try:
                        fx = self.registry.fx(s)
                    except KeyError:
                        log.warning("缺少 %s 汇率，市值按 1:1 计入", s)
                mv += p.market_value * fx
            total = cash + mv
        return AccountSnapshot(cash=cash, total_value=total, currency=cur)

    def query_positions(self) -> dict[str, Position]:
        return self._load_state()[3]

    # ---------------------------------------------------------------- orders
    def _write(self, row: dict) -> None:
        new = not self.orders_path.exists() or self.orders_path.stat().st_size == 0
        with self.orders_path.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_ORDER_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    def _row(self, action: str, oid: str, order: Order, status: str) -> dict:
        return dict(time=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), order_id=oid,
                    action=action, symbol=order.symbol,
                    code=_CODE_FORMATS[self.code_format](order.symbol),
                    direction=order.direction.value, quantity=order.quantity,
                    price="" if order.price is None else order.price,
                    tag=order.tag, status=status)

    def send_order(self, order: Order) -> str:
        self._seq += 1
        oid = f"M{self._seq:06d}"
        self._write(self._row("new", oid, order, "待人工下单"))
        self.open_orders[oid] = order
        log.info("订单已写入 %s: %s %s %d @ %s", self.orders_path, order.direction.value,
                 order.symbol, order.quantity, order.price or "MKT")
        return oid

    def cancel_order(self, broker_order_id: str) -> None:
        order = self.open_orders.pop(broker_order_id, None)
        if order is None:
            raise GatewayError(f"未找到订单 {broker_order_id}")
        self._write(self._row("cancel", broker_order_id, order, "取消"))

    def query_open_orders(self) -> list[dict]:
        return [dict(order_id=k, symbol=o.symbol, direction=o.direction.value,
                     quantity=o.quantity, price=o.price, filled=0, status="manual")
                for k, o in self.open_orders.items()]


class PtradeGateway(ManualGateway):
    """Ptrade (恒生) cannot be driven from the local PC: strategies run on the
    broker's server and the API is only available inside that sandbox.

    This gateway therefore *exports* orders as a CSV with Ptrade codes
    (``600519.SS``). Upload it to the Ptrade 研究 directory and let a small
    Ptrade strategy place them (see :data:`PTRADE_STRATEGY_TEMPLATE`). Account
    data is read from ``state_path`` exactly like :class:`ManualGateway`.
    """
    name = "ptrade"
    markets = ("CN",)

    def __init__(self, orders_path: str = "ptrade_orders.csv", **kw: Any):
        kw.setdefault("code_format", "ptrade")
        super().__init__(orders_path=orders_path, **kw)

    def connect(self) -> None:
        log.warning("Ptrade 策略运行在券商服务器上，无法从本机直接下单；订单将导出到 %s，"
                    "请上传到 Ptrade 研究目录并由 Ptrade 策略读取执行", self.orders_path)
        super().connect()


PTRADE_STRATEGY_TEMPLATE = '''\
# Ptrade strategy: place orders exported by quantum_analyze (python 3.5 compatible)
import csv, datetime, os

def initialize(context):
    g.done_day = None
    run_daily(context, place_orders, time="09:35")

def place_orders(context):
    today = datetime.date.today().strftime("%Y-%m-%d")
    path = os.path.join(get_research_path(), "ptrade_orders.csv")
    if g.done_day == today or not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cancelled = set(r["order_id"] for r in rows if r["action"] == "cancel")
    for r in rows:
        if r["action"] != "new" or r["order_id"] in cancelled or r["time"][:10] != today:
            continue
        amount = int(r["quantity"]) * (1 if r["direction"] == "buy" else -1)
        if r["price"]:
            order(r["code"], amount, limit_price=float(r["price"]))
        else:
            order(r["code"], amount)
    g.done_day = today

def handle_data(context, data):
    pass
'''


def export_ptrade_orders(orders: Iterable[Order], path: str | Path) -> Path:
    """Write ``orders`` to a fresh Ptrade-format CSV and return its path."""
    path = Path(path)
    if path.exists():
        path.unlink()
    gw = ManualGateway(orders_path=str(path), code_format="ptrade")
    gw.connect()
    for o in orders:
        gw.send_order(o)
    return path


# =========================================================================
# local simulation
# =========================================================================
class SimGateway(BaseGateway):
    """Gateway over the built-in :class:`SimBroker` (local simulated matching).

    Lets the live path (``LiveAccount`` → gateway) run offline and serves as
    paper trading without any broker account. Orders are queued in the
    SimBroker; call :meth:`match` with the next bars to fill them. With
    ``state_path`` the broker state is loaded on connect and saved after every
    match and on close.
    """
    name = "sim"
    paper = True

    def __init__(self, initial_cash: float = 1_000_000.0, registry: Optional[MarketRegistry] = None,
                 broker: Optional[SimBroker] = None, state_path: Optional[str] = None):
        self.registry = registry or (broker.registry if broker else MarketRegistry())
        self.broker = broker or SimBroker(initial_cash, self.registry)
        self.state_path = Path(state_path) if state_path else None

    def connect(self) -> None:
        if self.state_path and self.state_path.exists():
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.broker = SimBroker.from_dict(d, self.registry)

    def close(self) -> None:
        self.save()

    def save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.broker.to_dict(), ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    def query_account(self) -> AccountSnapshot:
        return AccountSnapshot(cash=self.broker.cash, total_value=self.broker.total_value,
                               currency=self.registry.base_currency)

    def query_positions(self) -> dict[str, Position]:
        return {s: deepcopy(p) for s, p in self.broker.positions.items()}

    def send_order(self, order: Order) -> str:
        self.broker.submit(order)
        if order.status == OrderStatus.REJECTED:
            raise GatewayError(order.reason)
        return order.order_id

    def cancel_order(self, broker_order_id: str) -> None:
        for o in list(self.broker.pending):
            if o.order_id == broker_order_id:
                self.broker.cancel(o)
                return
        raise GatewayError(f"未找到挂单 {broker_order_id}")

    def query_open_orders(self) -> list[dict]:
        return [dict(order_id=o.order_id, symbol=o.symbol, direction=o.direction.value,
                     quantity=o.quantity, price=o.price, filled=o.filled_qty, status=o.status.value)
                for o in self.broker.pending]

    # ---- simulation passthroughs
    def on_day_start(self, dt: pd.Timestamp) -> None:
        self.broker.on_day_start(pd.Timestamp(dt))

    def update_prices(self, bars: dict[str, Bar]) -> None:
        self.broker.update_prices(bars)

    def match(self, bars: dict[str, Bar], price_field: str = "open") -> list[Trade]:
        if self.broker.current_dt is None and bars:   # trades need a timestamp to persist
            self.broker.set_time(max(b.datetime for b in bars.values()))
        trades = self.broker.match(bars, price_field)
        self.broker.update_prices(bars)
        self.save()
        return trades


# =========================================================================
# registry / factory
# =========================================================================
GATEWAYS: dict[str, type[BaseGateway]] = {
    "ibkr": IBGateway,
    "futu": FutuGateway,
    "tiger": TigerGateway,
    "longbridge": LongbridgeGateway,
    "qmt": QMTGateway,
    "easytrader": EasyTraderGateway,
    "ptrade": PtradeGateway,
    "manual": ManualGateway,
    "sim": SimGateway,
}

_ALIASES = {"ib": "ibkr", "interactive_brokers": "ibkr", "tws": "ibkr", "moomoo": "futu",
            "longport": "longbridge", "miniqmt": "qmt", "xtquant": "qmt", "ths": "easytrader",
            "csv": "manual", "paper": "sim", "local": "sim"}


def build_gateway(name: str, **options: Any) -> BaseGateway:
    """Instantiate a gateway by name (see :data:`GATEWAYS`); not yet connected."""
    key = _ALIASES.get(name.strip().lower(), name.strip().lower())
    if key not in GATEWAYS:
        raise GatewayError(f"未知网关 {name!r}，可选: {', '.join(GATEWAYS)}")
    return GATEWAYS[key](**options)
