"""Market rules and trading costs for A-shares, Hong Kong and US stocks.

Symbols follow the ``CODE.MARKET`` convention: ``600519.SH``, ``000001.SZ``,
``430047.BJ``, ``00700.HK``, ``AAPL.US``. A bare 6-digit code is treated as an
A-share and its exchange is inferred.

Cost defaults are approximations of common retail brokers in 2024-2026
(A-share 万2.5 min 5 CNY; Futu-like HK/US schedules). Every field can be
overridden from the config file, so plug in your own broker's fee sheet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass
class CostModel:
    commission_rate: float = 0.0       # proportional commission, both sides
    commission_per_share: float = 0.0  # per-share commission (US style)
    min_commission: float = 0.0        # minimum commission per order
    per_order_fee: float = 0.0         # fixed platform fee per order
    exchange_fee_rate: float = 0.0     # exchange / transfer / levy fees, both sides
    stamp_tax_rate: float = 0.0        # stamp duty
    stamp_tax_on_buy: bool = False     # HK charges stamp duty on both sides
    sell_fee_rate: float = 0.0         # sell-only regulatory fees (e.g. US SEC fee)
    slippage: float = 0.0005           # proportional slippage against you

    def fees(self, is_buy: bool, amount: float, quantity: int = 0) -> tuple[float, float]:
        """Return (commission + fees, tax) for a fill, in the market's currency."""
        if amount <= 0:
            return 0.0, 0.0
        commission = amount * self.commission_rate + quantity * self.commission_per_share
        commission = max(commission, self.min_commission) + self.per_order_fee
        commission += amount * self.exchange_fee_rate
        if not is_buy:
            commission += amount * self.sell_fee_rate
        tax = amount * self.stamp_tax_rate if (not is_buy or self.stamp_tax_on_buy) else 0.0
        return commission, tax


@dataclass
class MarketRules:
    lot_size: int = 100                # buy orders must be multiples of this
    t_plus: int = 1                    # 1 = shares bought today can be sold next day
    enforce_price_limit: bool = True
    tick: float = 0.01
    max_volume_ratio: float = 0.25     # max share of a bar's volume one order may take; 0 = off
    limit_rules: list[tuple[str, float]] = field(default_factory=list)  # code prefix -> pct
    default_limit: float = 0.10
    st_symbols: set[str] = field(default_factory=set)   # ST stocks use 5%
    lot_sizes: dict[str, int] = field(default_factory=dict)  # per-symbol lot override (HK)

    def lot_for(self, symbol: str) -> int:
        return self.lot_sizes.get(symbol, self.lot_size)

    def limit_pct(self, symbol: str) -> float:
        if symbol in self.st_symbols:
            return 0.05
        code = symbol.split(".")[0]
        for prefix, pct in self.limit_rules:
            if code.startswith(prefix):
                return pct
        return self.default_limit

    def limit_prices(self, symbol: str, pre_close: float) -> tuple[float, float]:
        if (not self.enforce_price_limit or pre_close is None
                or math.isnan(pre_close) or pre_close <= 0):
            return 0.0, math.inf
        pct = self.limit_pct(symbol)
        return round(pre_close * (1 - pct) + 1e-9, 2), round(pre_close * (1 + pct) + 1e-9, 2)

    def round_lot(self, symbol: str, qty: float) -> int:
        lot = self.lot_for(symbol)
        if lot <= 1:
            return int(math.floor(qty + 1e-9))
        return int(math.floor(qty / lot + 1e-9)) * lot


@dataclass
class MarketSpec:
    name: str
    currency: str
    rules: MarketRules
    cost: CostModel


def a_share_spec() -> MarketSpec:
    return MarketSpec(
        name="CN", currency="CNY",
        rules=MarketRules(
            lot_size=100, t_plus=1, enforce_price_limit=True, tick=0.01,
            limit_rules=[("688", 0.20), ("689", 0.20), ("300", 0.20), ("301", 0.20),
                         ("8", 0.30), ("4", 0.30), ("92", 0.30)],
            default_limit=0.10,
        ),
        cost=CostModel(commission_rate=0.00025, min_commission=5.0,
                       exchange_fee_rate=0.00001, stamp_tax_rate=0.0005),
    )


def hk_spec() -> MarketSpec:
    return MarketSpec(
        name="HK", currency="HKD",
        rules=MarketRules(lot_size=100, t_plus=0, enforce_price_limit=False, tick=0.001),
        # commission 0.03% (min 3) + platform 15/order; stamp 0.1% both sides;
        # trading fee + SFC levy + AFRC + settlement ~0.0105%
        cost=CostModel(commission_rate=0.0003, min_commission=3.0, per_order_fee=15.0,
                       exchange_fee_rate=0.000105, stamp_tax_rate=0.001,
                       stamp_tax_on_buy=True),
    )


def us_spec() -> MarketSpec:
    return MarketSpec(
        name="US", currency="USD",
        rules=MarketRules(lot_size=1, t_plus=0, enforce_price_limit=False, tick=0.01),
        # ~0.0099 USD/share (commission + platform), min 1.98; SEC/TAF on sells
        cost=CostModel(commission_per_share=0.0099, min_commission=1.98,
                       sell_fee_rate=0.0000278),
    )


def market_of(symbol: str) -> str:
    """Return CN / HK / US for a symbol."""
    if "." in symbol:
        suffix = symbol.rsplit(".", 1)[1].upper()
        if suffix in ("SH", "SZ", "BJ", "SS"):
            return "CN"
        if suffix == "HK":
            return "HK"
        return "US"
    code = symbol.strip()
    if code.isdigit() and len(code) == 6:
        return "CN"
    if code.isdigit() and len(code) == 5:
        return "HK"
    return "US"


def normalize_symbol(symbol: str) -> str:
    """``600519`` -> ``600519.SH``; ``700`` stays as given unless suffixed."""
    s = symbol.strip().upper()
    if "." in s:
        return s
    if s.isdigit() and len(s) == 6:
        if s.startswith(("6", "5", "9")):
            return f"{s}.SH"
        if s.startswith(("8", "4", "92")):
            return f"{s}.BJ"
        return f"{s}.SZ"
    if s.isdigit() and len(s) <= 5:
        return f"{s.zfill(5)}.HK"
    return f"{s}.US"


class MarketRegistry:
    """Resolves the rules/costs/currency for any symbol, plus FX into base currency."""

    def __init__(self, base_currency: str = "CNY",
                 fx_rates: dict[str, float] | None = None,
                 specs: dict[str, MarketSpec] | None = None):
        self.base_currency = base_currency
        self.specs = specs or {"CN": a_share_spec(), "HK": hk_spec(), "US": us_spec()}
        # how many units of base currency one unit of each currency is worth
        self.fx_rates = {base_currency: 1.0, **(fx_rates or {})}

    def spec(self, symbol: str) -> MarketSpec:
        return self.specs[market_of(symbol)]

    def fx(self, symbol: str) -> float:
        cur = self.spec(symbol).currency
        if cur not in self.fx_rates:
            raise KeyError(f"缺少汇率: {cur} -> {self.base_currency}，请在配置 [account.fx_rates] 中设置")
        return self.fx_rates[cur]

    def copy(self) -> "MarketRegistry":
        specs = {k: replace(v, rules=replace(v.rules), cost=replace(v.cost)) for k, v in self.specs.items()}
        return MarketRegistry(self.base_currency, dict(self.fx_rates), specs)
