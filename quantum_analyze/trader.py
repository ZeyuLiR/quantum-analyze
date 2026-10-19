"""Paper and live trading runners.

Both are *run once per bar* (e.g. once a day from a scheduler) and persist
their state in a JSON file between runs:

* **paper** – the built-in simulated broker. Orders produced on day T fill
  at day T+1's open on the next run, with the same A-share/HK/US rules and
  costs as the backtest. Needs no broker account.
* **live** – a broker gateway (IBKR, Futu, Tiger, Longbridge, miniQMT ...).
  Positions and cash are read from the broker; orders are sent immediately.
  ``dry_run`` (default true) only prints the orders, so you can check what it
  would do before letting it trade real money. Use the broker's own paper
  account (IBKR paper port, Futu SIMULATE) for a realistic rehearsal.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import AppConfig
from .core.broker import SimBroker
from .engine import Engine
from .risk import RiskManager

log = logging.getLogger("quantum_analyze")


def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class PaperTrader:
    def __init__(self, cfg: AppConfig, state_file: Optional[str] = None):
        self.cfg = cfg
        self.state_path = cfg.resolve(state_file) if state_file else cfg.state_path("paper")

    def run(self, data: Optional[dict] = None, until: Optional[str] = None) -> dict:
        cfg = self.cfg
        state = _load_state(self.state_path)
        registry = cfg.build_registry()
        if data is None:
            warm = cfg.trading.get("history_start") or str(
                (pd.Timestamp.today() - pd.DateOffset(years=3)).date())
            data, _ = cfg.load_data(start=warm, end=until)
        broker = (SimBroker.from_dict(state["broker"], registry) if "broker" in state
                  else SimBroker(cfg.initial_cash, registry))
        risk = RiskManager(cfg.build_risk())
        if "risk" in state:
            risk.load(state["risk"])
        engine = Engine(cfg.build_strategy(), data, account=broker, risk=risk,
                        fill_at="next_open", registry=registry)
        engine.ctx.state = state.get("strategy_state", {})
        engine._last_day = pd.Timestamp(state["last_bar"]).normalize() if "last_bar" in state else None
        engine.prepare()

        cal = engine.calendar
        if until:
            cal = cal[cal <= pd.Timestamp(until) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
        if "last_bar" in state:
            new_bars = cal[cal > pd.Timestamp(state["last_bar"])]
        else:
            new_bars = cal[-1:]          # first run: start from the latest bar
            engine.strategy.on_start(engine.ctx)
        n_trades_before = len(broker.trades)
        for dt in new_bars:
            engine.step(dt)
        new_trades = broker.trades[n_trades_before:]

        history = state.get("equity", [])
        history += [[str(r[0]), r[3]] for r in engine.equity_rows]
        state.update({
            "mode": "paper", "strategy": engine.strategy.describe(),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "last_bar": str(new_bars[-1]) if len(new_bars) else state.get("last_bar"),
            "broker": broker.to_dict(), "risk": risk.to_dict(),
            "strategy_state": engine.ctx.state, "equity": history[-5000:],
            "logs": (state.get("logs", []) + engine.ctx.logs)[-500:],
        })
        _save_state(self.state_path, state)
        return {
            "bars_processed": len(new_bars),
            "last_bar": state["last_bar"],
            "total_value": broker.total_value, "cash": broker.cash,
            "positions": {s: p.quantity for s, p in broker.positions.items()},
            "new_trades": [t.to_dict() for t in new_trades],
            "next_orders": [o.to_dict() for o in broker.pending],
        }


class LiveTrader:
    def __init__(self, cfg: AppConfig, gateway=None, dry_run: Optional[bool] = None):
        self.cfg = cfg
        t = cfg.trading
        self.gateway_name = t.get("gateway", "sim")
        self.gateway = gateway
        self.dry_run = t.get("dry_run", True) if dry_run is None else dry_run
        self.state_path = cfg.state_path("live")

    def run(self, data: Optional[dict] = None) -> dict:
        from .gateway.base import LiveAccount
        from .gateway.brokers import build_gateway

        cfg = self.cfg
        state = _load_state(self.state_path)
        registry = cfg.build_registry()
        if data is None:
            warm = cfg.trading.get("history_start") or str(
                (pd.Timestamp.today() - pd.DateOffset(years=3)).date())
            data, _ = cfg.load_data(start=warm, refresh=True)
        if self.gateway is None:
            import inspect
            from .gateway.brokers import GATEWAYS
            opts = dict(cfg.trading.get("gateway_options", {}))
            cls = GATEWAYS.get(self.gateway_name)
            if cls is not None and "registry" in inspect.signature(cls.__init__).parameters:
                opts.setdefault("registry", registry)
            gw = build_gateway(self.gateway_name, **opts)
        else:
            gw = self.gateway
        gw.connect()
        try:
            account = LiveAccount(gw, registry, dry_run=self.dry_run,
                                  limit_offset=float(cfg.trading.get("limit_offset", 0.0)))
            risk = RiskManager(cfg.build_risk())
            if "risk" in state:
                risk.load(state["risk"])
            engine = Engine(cfg.build_strategy(), data, account=account, risk=risk, registry=registry)
            engine.ctx.state = state.get("strategy_state", {})
            account.price_ref = engine.ctx.price
            engine.prepare()
            last = engine.calendar[-1]
            if state.get("last_bar") == str(last) and not cfg.trading.get("allow_rerun", False):
                log.warning("最新K线 %s 已经处理过，跳过（设置 trading.allow_rerun=true 可强制重跑）", last)
                return {"skipped": True, "last_bar": str(last)}
            engine.step(last)
            orders = [o.to_dict() for o in account.orders]
            state.update({
                "mode": "live", "gateway": self.gateway_name, "dry_run": self.dry_run,
                "strategy": engine.strategy.describe(),
                "updated": datetime.now().isoformat(timespec="seconds"),
                "last_bar": str(last) if not self.dry_run else state.get("last_bar"),
                # a dry run must not advance strategy / risk state
                "risk": state.get("risk", {}) if self.dry_run else risk.to_dict(),
                "strategy_state": state.get("strategy_state", {}) if self.dry_run else engine.ctx.state,
                "account": {"cash": account.cash, "total_value": account.total_value,
                            "positions": {s: p.to_dict() for s, p in account.positions.items()}},
                "equity": (state.get("equity", []) + [[str(last), account.total_value]])[-5000:],
                "orders": (state.get("orders", []) + orders)[-1000:],
                "logs": (state.get("logs", []) + engine.ctx.logs)[-500:],
            })
            _save_state(self.state_path, state)
            return {"last_bar": str(last), "dry_run": self.dry_run, "orders": orders,
                    "total_value": account.total_value, "cash": account.cash}
        finally:
            gw.close()
