"""TOML configuration → engine components.

One file describes account, data, universe, strategy, risk, costs and the
trading mode, so the very same file drives backtest, paper and live runs.
See examples/*.toml.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Optional

from .core.rules import MarketRegistry, normalize_symbol
from .risk import RiskConfig
from .strategies.base import Strategy, get_strategy


@dataclass
class AppConfig:
    path: Optional[Path] = None
    account: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    universe: dict = field(default_factory=dict)
    backtest: dict = field(default_factory=dict)
    strategy: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)
    trading: dict = field(default_factory=dict)
    optimize: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        with open(p, "rb") as f:
            raw = tomllib.load(f)
        known = {f.name for f in fields(cls)} - {"path"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"配置文件中有未知段: {sorted(unknown)}")
        return cls(path=p, **raw)

    def resolve(self, rel: str) -> Path:
        """Paths in the config are relative to the config file's folder."""
        p = Path(rel)
        if p.is_absolute() or self.path is None:
            return p
        return self.path.parent / p

    def state_path(self, mode: str) -> Path:
        """Paper and live keep separate state files: ``state_file`` is the paper one and
        the live one gets a ``_live`` suffix unless ``live_state_file`` is given."""
        t = self.trading
        if mode == "paper":
            return self.resolve(t.get("state_file", "state/paper.json"))
        if "live_state_file" in t:
            return self.resolve(t["live_state_file"])
        if "state_file" in t:
            p = self.resolve(t["state_file"])
            return p.with_name(f"{p.stem}_live{p.suffix}")
        return self.resolve(f"state/live_{t.get('gateway', 'sim')}.json")

    # ---------------------------------------------------------------- builders
    @property
    def symbols(self) -> list[str]:
        return [normalize_symbol(s) for s in self.universe.get("symbols", [])]

    @property
    def benchmark(self) -> Optional[str]:
        b = self.universe.get("benchmark")
        return normalize_symbol(b) if b else None

    @property
    def initial_cash(self) -> float:
        return float(self.account.get("initial_cash", 1_000_000))

    def build_registry(self) -> MarketRegistry:
        reg = MarketRegistry(self.account.get("base_currency", "CNY"),
                             self.account.get("fx_rates", {}))
        for market, overrides in self.cost.items():
            spec = reg.specs[market.upper()]
            spec.cost = replace(spec.cost, **overrides)
        for market, overrides in self.rules.items():
            spec = reg.specs[market.upper()]
            o = dict(overrides)
            if "lot_sizes" in o:
                o["lot_sizes"] = {normalize_symbol(k): int(v) for k, v in o["lot_sizes"].items()}
            if "st_symbols" in o:
                o["st_symbols"] = {normalize_symbol(s) for s in o["st_symbols"]}
            spec.rules = replace(spec.rules, **o)
        return reg

    def build_strategy(self, **override: Any) -> Strategy:
        cls = get_strategy(self.strategy.get("name", "buy_hold"))
        params = {**self.strategy.get("params", {}), **override}
        for key in ("symbol", "safe_asset", "market_symbol"):
            if params.get(key):
                params[key] = normalize_symbol(params[key])
        return cls(**params)

    def build_risk(self) -> RiskConfig:
        r = dict(self.risk)
        if "blacklist" in r:
            r["blacklist"] = [normalize_symbol(s) for s in r["blacklist"]]
        return RiskConfig.from_dict(r)

    def build_data_manager(self):
        from .data.manager import DataManager
        cfg = dict(self.data)
        for key in ("cache_dir",):
            if cfg.get(key):
                cfg[key] = str(self.resolve(cfg[key]))
        opts = {k: dict(v) for k, v in cfg.get("options", {}).items()}
        if "csv" in opts and "path" in opts["csv"]:
            opts["csv"]["path"] = str(self.resolve(opts["csv"]["path"]))
        cfg["options"] = opts
        return DataManager.from_config(cfg)

    def load_data(self, start: Optional[str] = None, end: Optional[str] = None,
                  refresh: bool = False):
        """Return (data dict, benchmark close series or None)."""
        dm = self.build_data_manager()
        freq = self.data.get("frequency", "1d")
        kwargs = {"refresh": True} if refresh else {}
        syms = self.symbols
        extra = []
        for key in ("safe_asset", "market_symbol", "symbol"):
            s = self.strategy.get("params", {}).get(key)
            if s and normalize_symbol(s) not in syms:
                extra.append(normalize_symbol(s))
        data = dm.load(syms + extra, start, end, freq, **kwargs)
        bench = None
        if self.benchmark:
            try:
                bench = dm.load([self.benchmark], start, end, "1d", **kwargs)[self.benchmark]["close"]
            except Exception as e:  # benchmark is optional
                import logging
                logging.getLogger("quantum_analyze").warning("基准 %s 加载失败: %s", self.benchmark, e)
        return data, bench
