"""Command line interface: ``qa <command> -c config.toml``.

Commands
  strategies            list built-in strategies and their parameters
  backtest              run a backtest and write an HTML report
  optimize              grid search parameters (in-sample)
  walkforward           rolling walk-forward validation (out-of-sample)
  fetch                 download / refresh data into the local cache
  paper                 run the paper-trading account up to the latest bar
  live                  run one live cycle through a broker gateway
  schedule              run paper/live every trading day at a fixed time
  dashboard             local web dashboard for accounts and reports
  ui                    graphical interface in the browser (all of the above)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _parse_grid(text: str) -> dict[str, list]:
    """``"fast=5,10;slow=30,60"`` → ``{"fast": [5, 10], "slow": [30, 60]}``."""
    grid = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        key, vals = part.split("=", 1)
        grid[key.strip()] = [_auto(v) for v in vals.split(",")]
    return grid


def _auto(v: str):
    v = v.strip()
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def cmd_strategies(args) -> None:
    from .strategies.base import STRATEGY_REGISTRY, get_strategy
    get_strategy("buy_hold")
    for name, cls in sorted(STRATEGY_REGISTRY.items()):
        print(f"{name:22s} {cls.title}")
        print(f"{'':22s} 默认参数: {cls.default_params}")
        if cls.param_grid:
            print(f"{'':22s} 建议寻优: {cls.param_grid}")


def _bt_kwargs(cfg) -> dict:
    return dict(initial_cash=cfg.initial_cash, fill_at=cfg.backtest.get("fill_at", "next_open"),
                risk=cfg.build_risk(), registry=cfg.build_registry())


def cmd_backtest(args) -> None:
    from .config import AppConfig
    from .engine import run_backtest
    cfg = AppConfig.load(args.config)
    start, end = args.start or cfg.backtest.get("start"), args.end or cfg.backtest.get("end")
    warmup = cfg.backtest.get("warmup_start")
    data, bench = cfg.load_data(start=warmup or _warmup(start), end=end, refresh=args.refresh)
    params = _parse_grid(args.params) if args.params else {}
    strategy = cfg.build_strategy(**{k: v[0] for k, v in params.items()})
    res = run_backtest(strategy, data, start=start, end=end, benchmark=bench, **_bt_kwargs(cfg))
    print(res.summary())
    out = cfg.resolve(args.out or cfg.backtest.get("output", f"reports/{strategy.name}"))
    print(f"\n报告: {Path(res.save(str(out))).resolve()}")


def _warmup(start):
    if not start:
        return None
    return str((pd.Timestamp(start) - pd.DateOffset(years=1)).date())


def cmd_optimize(args) -> None:
    from .config import AppConfig
    from .optimize import grid_search
    cfg = AppConfig.load(args.config)
    start, end = args.start or cfg.backtest.get("start"), args.end or cfg.backtest.get("end")
    data, _ = cfg.load_data(start=_warmup(start), end=end)
    strategy = cfg.build_strategy()
    grid = _parse_grid(args.grid) if args.grid else (cfg.optimize.get("grid") or type(strategy).param_grid)
    table = grid_search(type(strategy), data, grid, start=start, end=end,
                        objective=args.objective, base_params=cfg.strategy.get("params", {}),
                        **_bt_kwargs(cfg))
    pd.set_option("display.width", 200)
    print(table.head(args.top).to_string())
    if args.out:
        out = cfg.resolve(args.out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "optimize.csv", index=False)
        print(f"\n结果: {(out / 'optimize.csv').resolve()}")
    print("\n提示: 样本内寻优必然过拟合，请用 `qa walkforward` 检验样本外表现，并优先选择“参数高原”而不是单个尖峰。")


def cmd_walkforward(args) -> None:
    from .analysis.metrics import format_metrics
    from .config import AppConfig
    from .optimize import walk_forward
    cfg = AppConfig.load(args.config)
    start, end = args.start or cfg.backtest.get("start"), args.end or cfg.backtest.get("end")
    data, _ = cfg.load_data(start=_warmup(start), end=end)
    strategy = cfg.build_strategy()
    grid = _parse_grid(args.grid) if args.grid else (cfg.optimize.get("grid") or type(strategy).param_grid)
    kw = _bt_kwargs(cfg)
    out = walk_forward(type(strategy), data, grid, train_months=args.train, test_months=args.test,
                       objective=args.objective, start=start, end=end,
                       base_params=cfg.strategy.get("params", {}), **kw)
    pd.set_option("display.width", 200)
    print(out["windows"].to_string())
    print("\n样本外拼接表现:")
    for k, v in format_metrics(out["oos_metrics"]):
        print(f"  {k:<10s} {v}")
    if args.out:
        from .engine import write_json
        d = cfg.resolve(args.out)
        d.mkdir(parents=True, exist_ok=True)
        out["windows"].to_csv(d / "walkforward.csv", index=False)
        write_json(d / "walkforward.json", {"windows": out["windows"].to_dict("records"),
                                            "oos": format_metrics(out["oos_metrics"])})
        print(f"\n结果: {(d / 'walkforward.json').resolve()}")


def cmd_fetch(args) -> None:
    from .config import AppConfig
    cfg = AppConfig.load(args.config)
    data, bench = cfg.load_data(start=args.start or cfg.backtest.get("start"), end=args.end,
                                refresh=args.refresh)
    for sym, df in data.items():
        print(f"{sym:12s} {len(df):6d} 根K线  {df.index[0].date()} ~ {df.index[-1].date()}")
    if bench is not None:
        print(f"基准 {cfg.benchmark}: {len(bench)} 根")


def cmd_paper(args) -> None:
    from .config import AppConfig
    from .trader import PaperTrader
    cfg = AppConfig.load(args.config)
    out = PaperTrader(cfg, args.state).run(until=args.until)
    _print_run(out)


def cmd_live(args) -> None:
    from .config import AppConfig
    from .trader import LiveTrader
    cfg = AppConfig.load(args.config)
    dry = None
    if args.send:
        dry = False
    elif args.dry_run:
        dry = True
    if dry is False and cfg.trading.get("dry_run", True):
        ans = input("即将向券商发送真实委托（trading.dry_run=true 已被 --send 覆盖）。输入 YES 确认: ")
        if ans.strip() != "YES":
            print("已取消")
            return
    out = LiveTrader(cfg, dry_run=dry).run()
    _print_run(out)


def _print_run(out: dict) -> None:
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def cmd_schedule(args) -> None:
    """Blocking loop: run paper/live once per weekday at HH:MM local time."""
    hh, mm = map(int, args.at.split(":"))
    print(f"调度已启动：每个工作日 {args.at} 运行 {args.mode}（Ctrl+C 退出）")
    while True:
        now = datetime.now()
        nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        time.sleep(max((nxt - datetime.now()).total_seconds(), 1))
        try:
            (cmd_paper if args.mode == "paper" else cmd_live)(args)
        except Exception as e:  # keep the scheduler alive
            logging.getLogger("quantum_analyze").exception("运行失败: %s", e)


def cmd_ui(args) -> None:
    from .webui.server import serve
    serve(args.root, port=args.port, open_browser=not args.no_browser)


def cmd_dashboard(args) -> None:
    from .dashboard import serve
    root = Path(args.config).parent if args.config else Path(args.root)
    serve(root, port=args.port)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="qa", description="中低频量化交易：回测 / 模拟盘 / 实盘")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, config=True):
        sp = sub.add_parser(name, help=fn.__doc__)
        if config:
            sp.add_argument("-c", "--config", required=True, help="TOML 配置文件")
        sp.set_defaults(fn=fn)
        return sp

    add("strategies", cmd_strategies, config=False)
    sp = add("backtest", cmd_backtest)
    sp.add_argument("--start"); sp.add_argument("--end"); sp.add_argument("--out")
    sp.add_argument("--params", help='覆盖参数，如 "fast=5;slow=20"')
    sp.add_argument("--refresh", action="store_true", help="忽略缓存重新下载")
    for name, fn in (("optimize", cmd_optimize), ("walkforward", cmd_walkforward)):
        sp = add(name, fn)
        sp.add_argument("--start"); sp.add_argument("--end")
        sp.add_argument("--grid", help='参数网格，如 "fast=5,10,20;slow=30,60"')
        sp.add_argument("--out", help="把结果表保存到该文件夹（相对于配置文件）")
        sp.add_argument("--objective", default="sharpe",
                        choices=["sharpe", "annual_return", "calmar", "sortino"])
        if name == "optimize":
            sp.add_argument("--top", type=int, default=20)
        else:
            sp.add_argument("--train", type=int, default=36, help="训练窗口(月)")
            sp.add_argument("--test", type=int, default=12, help="测试窗口(月)")
    sp = add("fetch", cmd_fetch)
    sp.add_argument("--start"); sp.add_argument("--end"); sp.add_argument("--refresh", action="store_true")
    sp = add("paper", cmd_paper)
    sp.add_argument("--state"); sp.add_argument("--until", help="只处理到该日期（复盘/补跑用）")
    sp = add("live", cmd_live)
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="只打印委托，不发送")
    g.add_argument("--send", action="store_true", help="真实发送委托（需二次确认）")
    sp = add("schedule", cmd_schedule)
    sp.add_argument("--mode", choices=["paper", "live"], default="paper")
    sp.add_argument("--at", default="15:30", help="每日运行时间 HH:MM")
    sp.add_argument("--state"); sp.set_defaults(until=None, send=False, dry_run=False)
    sp = sub.add_parser("ui", help="图形界面（浏览器中操作）")
    sp.add_argument("--root", default=".", help="工作目录（默认当前目录）")
    sp.add_argument("--port", type=int, default=8501)
    sp.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    sp.set_defaults(fn=cmd_ui)
    sp = sub.add_parser("dashboard", help="本地网页看板")
    sp.add_argument("-c", "--config"); sp.add_argument("--root", default=".")
    sp.add_argument("--port", type=int, default=8050); sp.set_defaults(fn=cmd_dashboard)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    from .data.base import DataSourceError
    try:
        args.fn(args)
    except (DataSourceError, FileNotFoundError, ValueError, KeyError) as e:
        if args.verbose:
            raise
        print(f"\n出错了: {e}\n（加 -v 参数可查看详细报错）", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
