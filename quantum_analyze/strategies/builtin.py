"""Built-in medium/low-frequency strategies.

See docs/策略说明书.md for the logic, parameters, suitable markets and risks
of each one.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..core.context import Context
from ..risk import position_size_by_atr
from .base import Strategy, register


class SignalStrategy(Strategy):
    """Per-symbol long/flat timing strategy with equal-weight slots.

    Subclasses implement ``entry`` and ``exit``; capital is split into
    ``max_positions`` slots (defaults to the size of the universe).
    """

    def slot_weight(self, ctx: Context) -> float:
        n = self.params.get("max_positions") or len(ctx.universe)
        return 1.0 / max(n, 1)

    def entry(self, ctx: Context, sym: str) -> bool:
        raise NotImplementedError

    def exit(self, ctx: Context, sym: str) -> bool:
        raise NotImplementedError

    def target_weight(self, ctx: Context, sym: str) -> float:
        return self.slot_weight(ctx)

    def on_bar(self, ctx: Context) -> None:
        held = set(ctx.positions)
        for sym in ctx.universe:
            if not ctx.is_trading(sym):
                continue
            if sym in held:
                if self.exit(ctx, sym):
                    ctx.order_target_shares(sym, 0, tag="exit")
        slots = (self.params.get("max_positions") or len(ctx.universe)) - len(
            [s for s in ctx.positions if ctx.position(s).quantity > 0 and not ctx.account.pending_for(s)])
        candidates = [s for s in ctx.universe
                      if s not in held and ctx.is_trading(s) and self.entry(ctx, s)]
        candidates.sort(key=lambda s: -(self.rank(ctx, s) or 0))
        for sym in candidates[:max(slots, 0)]:
            ctx.order_target_percent(sym, self.target_weight(ctx, sym) * 0.99, tag="entry")

    def rank(self, ctx: Context, sym: str) -> float:
        """Priority when there are more signals than free slots."""
        return self.value(ctx, sym, "mom") or 0.0

    def _base_ind(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"close": df["close"], "mom": ta.momentum(df["close"], 20)}, index=df.index)


# ============================================================================ trend
@register("dual_ma")
class DualMA(SignalStrategy):
    title = "双均线趋势跟踪"
    default_params = {"fast": 10, "slow": 30, "trend_filter": 0, "max_positions": 0}
    param_grid = {"fast": [5, 10, 20], "slow": [30, 60, 120]}

    def prepare(self, data):
        for sym, df in data.items():
            ind = self._base_ind(df)
            ind["fast"] = ta.sma(df["close"], self.fast)
            ind["slow"] = ta.sma(df["close"], self.slow)
            ind["trend"] = ta.sma(df["close"], self.trend_filter) if self.trend_filter else np.nan
            self.ind[sym] = ind

    def entry(self, ctx, sym):
        f, s = self.value(ctx, sym, "fast"), self.value(ctx, sym, "slow")
        f1, s1 = self.value(ctx, sym, "fast", 1), self.value(ctx, sym, "slow", 1)
        if None in (f, s, f1, s1):
            return False
        golden = f > s and f1 <= s1
        if self.trend_filter:
            t = self.value(ctx, sym, "trend")
            return golden and t is not None and ctx.price(sym) > t
        return golden

    def exit(self, ctx, sym):
        f, s = self.value(ctx, sym, "fast"), self.value(ctx, sym, "slow")
        return f is not None and s is not None and f < s


@register("macd_trend")
class MACDTrend(SignalStrategy):
    title = "MACD 趋势 + 长期均线过滤"
    default_params = {"fast": 12, "slow": 26, "signal": 9, "filter_ma": 120, "max_positions": 0}
    param_grid = {"fast": [8, 12], "slow": [26, 34], "filter_ma": [60, 120, 250]}

    def prepare(self, data):
        for sym, df in data.items():
            ind = self._base_ind(df)
            ind = ind.join(ta.macd(df["close"], self.fast, self.slow, self.signal))
            ind["ma"] = ta.sma(df["close"], self.filter_ma)
            self.ind[sym] = ind

    def entry(self, ctx, sym):
        h, h1, ma = (self.value(ctx, sym, "hist"), self.value(ctx, sym, "hist", 1),
                     self.value(ctx, sym, "ma"))
        if None in (h, h1, ma):
            return False
        return h > 0 >= h1 and ctx.price(sym) > ma

    def exit(self, ctx, sym):
        h, ma = self.value(ctx, sym, "hist"), self.value(ctx, sym, "ma")
        return (h is not None and h < 0) or (ma is not None and ctx.price(sym) < ma)


@register("turtle")
class Turtle(SignalStrategy):
    title = "海龟突破（唐奇安通道 + ATR 仓位 + ATR 止损）"
    default_params = {"entry_n": 20, "exit_n": 10, "atr_n": 20, "risk_per_trade": 0.01,
                      "stop_atr": 2.0, "max_weight": 0.25, "max_positions": 0}
    param_grid = {"entry_n": [20, 55], "exit_n": [10, 20], "stop_atr": [2.0, 3.0]}

    def prepare(self, data):
        for sym, df in data.items():
            ind = self._base_ind(df)
            up = ta.donchian(df, self.entry_n)["upper"]
            dn = ta.donchian(df, self.exit_n)["lower"]
            ind["upper"], ind["lower"], ind["atr"] = up, dn, ta.atr(df, self.atr_n)
            self.ind[sym] = ind

    def entry(self, ctx, sym):
        up = self.value(ctx, sym, "upper")
        return up is not None and ctx.price(sym) > up

    def exit(self, ctx, sym):
        lo = self.value(ctx, sym, "lower")
        stop = ctx.state.get("turtle_stop", {}).get(sym)
        px = ctx.price(sym)
        hit = (lo is not None and px < lo) or (stop is not None and px < stop)
        if hit:
            ctx.state.get("turtle_stop", {}).pop(sym, None)
        return hit

    def target_weight(self, ctx, sym):
        atr, px = self.value(ctx, sym, "atr"), ctx.price(sym)
        if atr is None:
            return 0.0
        ctx.state.setdefault("turtle_stop", {})[sym] = px - self.stop_atr * atr
        w = position_size_by_atr(ctx.total_value, self.risk_per_trade, atr, self.stop_atr, px)
        return min(w, self.max_weight, self.slot_weight(ctx) * 2)


# ============================================================================ mean reversion
@register("rsi_reversion")
class RSIReversion(SignalStrategy):
    title = "RSI 超卖反转（顺大势、逆小势）"
    default_params = {"rsi_n": 6, "buy_below": 20, "sell_above": 60, "trend_ma": 120,
                      "max_hold": 10, "max_positions": 0}
    param_grid = {"rsi_n": [2, 6, 14], "buy_below": [10, 20, 30], "sell_above": [50, 60, 70]}

    def prepare(self, data):
        for sym, df in data.items():
            ind = self._base_ind(df)
            ind["rsi"] = ta.rsi(df["close"], self.rsi_n)
            ind["ma"] = ta.sma(df["close"], self.trend_ma)
            self.ind[sym] = ind

    def entry(self, ctx, sym):
        r, ma = self.value(ctx, sym, "rsi"), self.value(ctx, sym, "ma")
        return r is not None and ma is not None and r < self.buy_below and ctx.price(sym) > ma

    def exit(self, ctx, sym):
        r = self.value(ctx, sym, "rsi")
        pos = ctx.position(sym)
        held = 0
        if pos.entry_date is not None:
            held = int(((ctx.calendar > pos.entry_date) & (ctx.calendar <= ctx.dt)).sum())
        return (r is not None and r > self.sell_above) or held >= self.max_hold

    def rank(self, ctx, sym):
        return -(self.value(ctx, sym, "rsi") or 50)


@register("bollinger_reversion")
class BollingerReversion(SignalStrategy):
    title = "布林带均值回归"
    default_params = {"n": 20, "k": 2.0, "trend_ma": 0, "max_positions": 0}
    param_grid = {"n": [20, 30], "k": [1.5, 2.0, 2.5]}

    def prepare(self, data):
        for sym, df in data.items():
            ind = self._base_ind(df).join(ta.bollinger(df["close"], self.n, self.k))
            ind["ma"] = ta.sma(df["close"], self.trend_ma) if self.trend_ma else np.nan
            self.ind[sym] = ind

    def entry(self, ctx, sym):
        lo, lo1 = self.value(ctx, sym, "lower"), self.value(ctx, sym, "lower", 1)
        if lo is None or lo1 is None:
            return False
        px = ctx.price(sym)
        prev = ctx.history(sym, "close", 2)
        if len(prev) < 2:
            return False
        # close re-enters the band from below: confirmation instead of catching a falling knife
        ok = prev.iloc[0] < lo1 and px > lo
        if self.trend_ma:
            ma = self.value(ctx, sym, "ma")
            ok = ok and ma is not None and px > ma
        return ok

    def exit(self, ctx, sym):
        mid = self.value(ctx, sym, "mid")
        return mid is not None and ctx.price(sym) >= mid

    def rank(self, ctx, sym):
        return -(self.value(ctx, sym, "pct_b") or 0)


# ============================================================================ rotation / cross-section
@register("momentum_rotation")
class MomentumRotation(Strategy):
    title = "动量轮动（ETF / 指数 / 个股）"
    default_params = {"lookback": 25, "top_k": 1, "score": "r2_slope", "rebalance": "W",
                      "abs_threshold": 0.0, "safe_asset": "", "skip": 0}
    param_grid = {"lookback": [20, 25, 60], "top_k": [1, 2, 3], "rebalance": ["W", "M"]}

    def prepare(self, data):
        for sym, df in data.items():
            c = df["close"]
            if self.score == "r2_slope":
                s = ta.trend_r2_slope(c, self.lookback)
            else:
                s = ta.momentum(c, self.lookback, self.skip)
            self.ind[sym] = pd.DataFrame({"score": s}, index=df.index)

    def on_bar(self, ctx):
        if not ctx.is_period_start(self.rebalance):
            return
        scores = self.cross_section(ctx, "score")
        if self.safe_asset:
            scores = scores.drop(self.safe_asset, errors="ignore")
        picks = scores[scores > self.abs_threshold].sort_values(ascending=False).head(self.top_k)
        weights = {s: 1.0 / self.top_k for s in picks.index}
        if self.safe_asset and len(weights) < self.top_k:
            weights[self.safe_asset] = 1.0 - sum(weights.values())
        ctx.rebalance(weights, min_change=0.05)


@register("multi_factor")
class MultiFactor(Strategy):
    title = "量价多因子选股"
    default_params = {
        "top_k": 10, "rebalance": "M",
        "w_momentum": 1.0, "w_lowvol": 1.0, "w_reversal": 0.5, "w_liquidity": 0.0,
        "mom_n": 120, "mom_skip": 20, "vol_n": 60, "rev_n": 5, "liq_n": 20,
        "market_ma": 0, "market_symbol": "",
    }
    param_grid = {"top_k": [5, 10, 20], "w_lowvol": [0.0, 1.0], "w_reversal": [0.0, 0.5]}

    def prepare(self, data):
        for sym, df in data.items():
            c = df["close"]
            amount = df["amount"] if "amount" in df else c * df.get("volume", 0)
            self.ind[sym] = pd.DataFrame({
                "momentum": ta.momentum(c, self.mom_n, self.mom_skip),
                "lowvol": -ta.volatility(c, self.vol_n),
                "reversal": -ta.momentum(c, self.rev_n),
                # prefer *less* crowded names (illiquidity premium)
                "liquidity": -np.log1p(amount.rolling(self.liq_n, min_periods=self.liq_n).mean()),
                "ma": ta.sma(c, self.market_ma) if self.market_ma else np.nan,
            }, index=df.index)

    def score(self, ctx) -> pd.Series:
        total = None
        for f in ("momentum", "lowvol", "reversal", "liquidity"):
            w = self.params[f"w_{f}"]
            if not w:
                continue
            z = ta.zscore_cross(self.cross_section(ctx, f)) * w
            total = z if total is None else total.add(z, fill_value=0)
        return pd.Series(dtype=float) if total is None else total.dropna()

    def on_bar(self, ctx):
        if not ctx.is_period_start(self.rebalance):
            return
        if self.market_ma and self.market_symbol:
            ma = self.value(ctx, self.market_symbol, "ma")
            if ma is not None and ctx.price(self.market_symbol) < ma:
                ctx.log("大盘在均线下方，空仓避险")
                ctx.close_all(tag="market_timing")
                return
        scores = self.score(ctx)
        if self.market_symbol:
            scores = scores.drop(self.market_symbol, errors="ignore")
        picks = scores.sort_values(ascending=False).head(self.top_k).index
        ctx.rebalance({s: 1.0 / self.top_k for s in picks}, min_change=0.02)


@register("risk_parity")
class RiskParity(Strategy):
    title = "风险平价资产配置（股/债/商品/海外）"
    default_params = {"vol_n": 60, "rebalance": "M", "target_vol": 0.0, "max_weight": 0.6}
    param_grid = {"vol_n": [20, 60, 120]}

    def prepare(self, data):
        for sym, df in data.items():
            self.ind[sym] = pd.DataFrame({"vol": ta.volatility(df["close"], self.vol_n)}, index=df.index)

    def on_bar(self, ctx):
        if not ctx.is_period_start(self.rebalance):
            return
        vol = self.cross_section(ctx, "vol")
        vol = vol[vol > 0]
        if vol.empty:
            return
        inv = 1 / vol
        w = inv / inv.sum()
        for _ in range(10):  # cap and redistribute
            over = w > self.max_weight
            if not over.any():
                break
            excess = (w[over] - self.max_weight).sum()
            w[over] = self.max_weight
            w[~over] += excess * w[~over] / w[~over].sum()
        if self.target_vol:
            port_vol = math.sqrt(((w * vol) ** 2).sum())   # ignores correlation: conservative
            w = w * min(self.target_vol / port_vol, 1.0) if port_vol > 0 else w
        ctx.rebalance(w.to_dict(), min_change=0.03)


@register("grid")
class Grid(Strategy):
    title = "网格交易（震荡市 ETF）"
    default_params = {"symbol": "", "center_n": 120, "step": 0.03, "levels": 5,
                      "base_weight": 0.5, "max_weight": 0.95}
    param_grid = {"step": [0.02, 0.03, 0.05], "levels": [4, 5, 8]}

    def prepare(self, data):
        for sym, df in data.items():
            self.ind[sym] = pd.DataFrame({"center": ta.sma(df["close"], self.center_n)}, index=df.index)

    def on_bar(self, ctx):
        sym = self.symbol or ctx.universe[0]
        center = self.value(ctx, sym, "center")
        if center is None or not ctx.is_trading(sym):
            return
        level = int(round(math.log(ctx.price(sym) / center) / math.log(1 + self.step)))
        level = max(min(level, self.levels), -self.levels)
        last = ctx.state.get("grid_level")
        if last == level:
            return
        ctx.state["grid_level"] = level
        # price above centre → hold less; below → hold more
        w = self.base_weight - level * (self.base_weight / self.levels)
        w = min(max(w, 0.0), self.max_weight)
        ctx.order_target_percent(sym, w, tag=f"grid{level:+d}")


@register("buy_hold")
class BuyAndHold(Strategy):
    title = "买入持有（基准对照）"
    default_params = {}

    def on_bar(self, ctx):
        if ctx.state.get("bought"):
            return
        syms = [s for s in ctx.universe if ctx.is_trading(s)]
        if syms:
            ctx.rebalance({s: 1.0 / len(ctx.universe) for s in syms})
            ctx.state["bought"] = len(syms) == len(ctx.universe)
