"""Local web UI: every ``qa`` operation behind buttons in the browser.

Standard library only. Listens on 127.0.0.1; every POST must carry the
per-process token embedded in the page (so other websites open in the same
browser cannot drive it). Long operations run as ``python -m quantum_analyze``
subprocesses whose output streams into the page.
"""
from __future__ import annotations

import importlib.util
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import tomllib
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import tomlw
from .meta import PARAM_LABELS, STRATEGY_INFO

STATIC = Path(__file__).parent / "static"

# pip package to install for each optional component (whitelist for the install button)
PACKAGES = {
    "akshare": ("akshare", "A股/港股/美股免费数据（推荐）"),
    "baostock": ("baostock", "A股免费数据（股票/指数，不含ETF）"),
    "tushare": ("tushare", "A股数据（需注册 Token）"),
    "yfinance": ("yfinance", "港股/美股免费数据（推荐）"),
    "ib_async": ("ib_async", "盈透 IBKR 行情与交易"),
    "futu": ("futu-api", "富途 OpenAPI 行情与交易"),
    "tigeropen": ("tigeropen", "老虎证券交易"),
    "longport": ("longport", "长桥证券交易"),
    "easytrader": ("easytrader", "同花顺客户端自动化（A股，兜底）"),
}
NOT_ON_PIP = {"xtquant": "miniQMT（随券商 QMT 客户端提供，无法 pip 安装）"}

DATA_SOURCES = ["akshare", "baostock", "tushare", "qmt", "futu", "yfinance", "ibkr", "alphavantage",
                "csv", "synthetic"]
GATEWAYS = {
    "sim": "内置模拟（无需账户）", "ibkr": "盈透 IBKR", "futu": "富途", "tiger": "老虎",
    "longbridge": "长桥", "qmt": "miniQMT（A股）", "easytrader": "easytrader（A股）",
    "ptrade": "Ptrade（导出委托）", "manual": "手工下单（导出CSV）",
}


class Job:
    def __init__(self, action: str, argv: list[str], cwd: Path, stdin: str = "", meta: dict | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.action, self.argv, self.cwd = action, argv, cwd
        self.meta = meta or {}
        self.lines: list[str] = []
        self.status = "running"
        self.started = time.time()
        self.ended: float | None = None
        self.result: dict = {}
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", env=env, creationflags=flags)
        if stdin:
            self.proc.stdin.write(stdin)
        self.proc.stdin.close()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))
            if len(self.lines) > 5000:
                del self.lines[:1000]
        code = self.proc.wait()
        if self.status == "running":
            self.status = "done" if code == 0 else "failed"
        self.ended = time.time()
        try:
            self.result = collect_result(self)
        except Exception as e:  # result is a convenience; the log is authoritative
            self.result = {"error": str(e)}

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.status = "stopped"
            self.proc.terminate()

    def to_dict(self, since: int = 0) -> dict:
        return {"id": self.id, "action": self.action, "status": self.status, "meta": self.meta,
                "started": self.started, "ended": self.ended, "total": len(self.lines),
                "lines": self.lines[since:], "result": self.result if self.status != "running" else {}}


def collect_result(job: Job) -> dict:
    out = job.meta.get("out_dir")
    if not out:
        return {}
    d = Path(out)
    if job.action == "backtest" and (d / "metrics.json").exists():
        return {"metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8")),
                "report": job.meta.get("report_url")}
    if job.action == "walkforward" and (d / "walkforward.json").exists():
        return json.loads((d / "walkforward.json").read_text(encoding="utf-8"))
    if job.action == "optimize" and (d / "optimize.csv").exists():
        import pandas as pd
        df = pd.read_csv(d / "optimize.csv").head(30)
        return {"columns": list(df.columns),
                "rows": json.loads(df.to_json(orient="values", double_precision=6))}
    return {}


class App:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.token = secrets.token_urlsafe(24)
        self.jobs: dict[str, Job] = {}
        (self.root / "configs").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def safe(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError("路径不在工作目录内")
        return p

    def rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.root).as_posix()

    def config_files(self) -> list[dict]:
        out = []
        for folder, kind in (("configs", "我的配置"), ("examples", "示例模板")):
            for f in sorted((self.root / folder).glob("*.toml")):
                out.append({"path": self.rel(f), "name": f.stem, "kind": kind})
        return out

    def meta(self) -> dict:
        from ..strategies.base import STRATEGY_REGISTRY, get_strategy
        get_strategy("buy_hold")
        strategies = []
        for name, cls in sorted(STRATEGY_REGISTRY.items(), key=lambda kv: STRATEGY_INFO.get(kv[0], {}).get("order", 99)):
            info = STRATEGY_INFO.get(name, {})
            strategies.append({"name": name, "title": cls.title, "params": cls.default_params,
                               "grid": cls.param_grid, "summary": info.get("summary", ""),
                               "fit": info.get("fit", ""), "risk": info.get("risk", "")})
        env = []
        for mod, (pkg, desc) in PACKAGES.items():
            ok = importlib.util.find_spec(mod) is not None
            if mod == "ib_async" and not ok:
                ok = importlib.util.find_spec("ib_insync") is not None
            env.append({"module": mod, "package": pkg, "desc": desc, "installed": ok, "pip": True})
        for mod, desc in NOT_ON_PIP.items():
            env.append({"module": mod, "package": "", "desc": desc,
                        "installed": importlib.util.find_spec(mod) is not None, "pip": False})
        from .. import __version__
        return {"version": __version__, "python": sys.version.split()[0], "root": str(self.root),
                "strategies": strategies, "param_labels": PARAM_LABELS, "env": env,
                "data_sources": DATA_SOURCES, "gateways": GATEWAYS, "configs": self.config_files(),
                "docs": [p.name for p in sorted((self.root / "docs").glob("*.md"))]}

    def load_config(self, rel: str) -> dict:
        p = self.safe(rel)
        with open(p, "rb") as f:
            data = tomllib.load(f)
        return {"path": self.rel(p), "name": p.stem, "config": data,
                "text": p.read_text(encoding="utf-8")}

    def save_config(self, body: dict) -> dict:
        name = "".join(c for c in body.get("name", "") if c.isalnum() or c in "-_") or "my_strategy"
        target = self.root / "configs" / f"{name}.toml"
        if "text" in body:
            text = body["text"]
            tomllib.loads(text)                       # validate before writing
        else:
            text = tomlw.dumps(body["config"], header="由图形界面保存。也可以用记事本直接编辑。")
            tomllib.loads(text)
        from ..config import AppConfig
        target.write_text(text, encoding="utf-8")
        try:
            cfg = AppConfig.load(target)
            cfg.build_strategy()
            cfg.build_registry()
            cfg.build_risk()
        except Exception as e:
            return {"path": self.rel(target), "warning": f"已保存，但配置有问题: {e}"}
        return {"path": self.rel(target)}

    def state(self, rel: str) -> dict:
        from ..config import AppConfig
        cfg = AppConfig.load(self.safe(rel))
        out = {}
        for mode in ("paper", "live"):
            path = cfg.state_path(mode)
            if path.exists():
                out[mode] = json.loads(path.read_text(encoding="utf-8"))
        out["trading"] = cfg.trading
        return out

    # ------------------------------------------------------------------ actions
    def run(self, body: dict) -> Job:
        action = body["action"]
        py = [sys.executable, "-u", "-m", "quantum_analyze"]
        if action == "install":
            mod = body.get("module")
            if mod not in PACKAGES:
                raise ValueError("不支持安装该组件")
            argv = [sys.executable, "-m", "pip", "install", PACKAGES[mod][0]]
            if body.get("mirror"):
                argv += ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
            job = Job(action, argv, self.root, meta={"module": mod})
            self.jobs[job.id] = job
            return job

        cfg_path = self.safe(body["config"])
        if not cfg_path.exists():
            raise ValueError("配置文件不存在，请先保存")
        cfg_arg = str(cfg_path)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        meta = {"config": self.rel(cfg_path)}
        extra: list[str] = []
        stdin = ""
        if body.get("start"):
            extra += ["--start", body["start"]]
        if body.get("end"):
            extra += ["--end", body["end"]]
        if action in ("backtest", "optimize", "walkforward"):
            out = cfg_path.parent / "reports" / f"{cfg_path.stem}-{action}-{stamp}"
            meta["out_dir"] = str(out)
            extra += ["--out", str(out)]
            if action == "backtest":
                meta["report_url"] = "/files/" + self.rel(out / "report.html")
                if body.get("refresh"):
                    extra.append("--refresh")
            elif body.get("grid"):
                extra += ["--grid", body["grid"]]
            if action == "walkforward":
                extra += ["--train", str(int(body.get("train", 36))), "--test", str(int(body.get("test", 12)))]
            argv = py + [action, "-c", cfg_arg] + extra
        elif action == "fetch":
            argv = py + ["fetch", "-c", cfg_arg] + extra + (["--refresh"] if body.get("refresh") else [])
        elif action == "paper":
            argv = py + ["paper", "-c", cfg_arg]
        elif action == "schedule":
            at = body.get("at", "15:30")
            if len(at) != 5 or at[2] != ":":
                raise ValueError("时间格式应为 HH:MM")
            argv = py + ["schedule", "-c", cfg_arg, "--mode", "paper", "--at", at]
        elif action == "live_dry":
            argv = py + ["live", "-c", cfg_arg, "--dry-run"]
        elif action == "live_send":
            if body.get("confirm") != "确认下单":
                raise ValueError("请在确认框中输入“确认下单”")
            argv = py + ["live", "-c", cfg_arg, "--send"]
            stdin = "YES\n"
        else:
            raise ValueError(f"未知操作 {action}")
        job = Job(action, argv, self.root, stdin=stdin, meta=meta)
        self.jobs[job.id] = job
        return job


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "quantum-analyze-ui"

        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    html = (STATIC / "index.html").read_text(encoding="utf-8")
                    return self._send(200, html.replace("__TOKEN__", app.token).encode("utf-8"),
                                      "text/html; charset=utf-8")
                if u.path == "/api/meta":
                    return self._json(app.meta())
                if u.path == "/api/config":
                    return self._json(app.load_config(q["path"]))
                if u.path == "/api/state":
                    return self._json(app.state(q["path"]))
                if u.path == "/api/job":
                    job = app.jobs.get(q.get("id", ""))
                    if not job:
                        return self._json({"error": "任务不存在"}, 404)
                    return self._json(job.to_dict(int(q.get("since", 0))))
                if u.path == "/api/jobs":
                    return self._json([{k: v for k, v in j.to_dict().items() if k != "lines"}
                                       for j in app.jobs.values()])
                if u.path == "/api/doc":
                    p = app.safe("docs/" + q["name"])
                    if p.suffix != ".md":
                        raise ValueError("只能读取文档")
                    return self._json({"name": p.name, "text": p.read_text(encoding="utf-8")})
                if u.path.startswith("/files/"):
                    from urllib.parse import unquote
                    p = app.safe(unquote(u.path[len("/files/"):]))
                    types = {".html": "text/html; charset=utf-8", ".csv": "text/csv; charset=utf-8",
                             ".json": "application/json; charset=utf-8"}
                    if p.suffix not in types or not p.is_file():
                        return self._send(404, b"not found", "text/plain")
                    return self._send(200, p.read_bytes(), types[p.suffix])
                if u.path == "/dashboard":
                    from ..dashboard import render
                    return self._send(200, render(app.root).encode("utf-8"), "text/html; charset=utf-8")
                return self._send(404, b"not found", "text/plain")
            except Exception as e:
                return self._json({"error": str(e)}, 400)

        def do_POST(self):
            if not self._host_ok() or self.headers.get("X-QA-Token") != app.token:
                return self._json({"error": "令牌无效，请刷新页面"}, 403)
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                u = urlparse(self.path)
                if u.path == "/api/config/save":
                    return self._json(app.save_config(body))
                if u.path == "/api/run":
                    return self._json(app.run(body).to_dict())
                if u.path == "/api/job/stop":
                    job = app.jobs.get(body.get("id", ""))
                    if job:
                        job.stop()
                    return self._json({"ok": True})
                return self._json({"error": "not found"}, 404)
            except Exception as e:
                return self._json({"error": str(e)}, 400)

    return Handler


def serve(root: str | Path = ".", host: str = "127.0.0.1", port: int = 8501,
          open_browser: bool = True) -> None:
    app = App(Path(root))
    httpd = None
    for p in range(port, port + 20):           # find a free port
        try:
            httpd = ThreadingHTTPServer((host, p), make_handler(app))
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        raise OSError("找不到可用端口")
    url = f"http://127.0.0.1:{port}/"
    print(f"图形界面已启动: {url}\n工作目录: {app.root}\n关闭本窗口即可退出。")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in app.jobs.values():
            if job.action == "schedule":
                job.stop()
