"""Tiny local web dashboard (standard library only).

``qa dashboard -c config.toml`` serves http://127.0.0.1:8050 showing every
paper/live state file and backtest report under the config's folder. It only
listens on localhost and is read-only.
"""
from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from .analysis.metrics import compute_metrics, format_metrics
from .analysis.report import CSS, line_chart


def _state_section(path: Path) -> str:
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"<h2>{html.escape(path.name)}</h2><p>读取失败: {html.escape(str(e))}</p>"
    mode = st.get("mode", "?")
    title = f"{'模拟盘' if mode == 'paper' else '实盘'} · {st.get('strategy', '')}"
    if st.get("dry_run"):
        title += "（dry-run）"
    out = [f"<h2>{html.escape(title)}</h2>",
           f"<p class='muted'>{html.escape(str(path))} · 更新于 {html.escape(str(st.get('updated')))}"
           f" · 最新K线 {html.escape(str(st.get('last_bar')))}</p>"]
    eq = pd.Series({pd.Timestamp(d): v for d, v in st.get("equity", [])}, dtype=float)
    if len(eq) > 1:
        m = compute_metrics(eq)
        keep = {"total_return", "annual_return", "max_drawdown", "sharpe", "final_value"}
        out.append('<div class="grid">' + "".join(
            f'<div class="kpi"><span>{k}</span><b>{v}</b></div>'
            for k, v in format_metrics({k: v for k, v in m.items() if k in keep})) + "</div>")
        out.append(line_chart({"账户净值": eq / eq.iloc[0]}, {"账户净值": "var(--a)"}, h=200))
    if "broker" in st:
        b = st["broker"]
        pos = b["positions"]
        out.append(f"<p>现金 {b['cash']:,.2f}</p>")
        rows = [(s, p["quantity"], p["sellable"] if "sellable" in p else p["quantity"] - p["today_bought"],
                 p["avg_cost"], p["last_price"]) for s, p in pos.items()]
        pending = b.get("pending", [])
    else:
        acct = st.get("account", {})
        out.append(f"<p>现金 {acct.get('cash', 0):,.2f} · 总资产 {acct.get('total_value', 0):,.2f}</p>")
        rows = [(s, p["quantity"], p["quantity"] - p.get("today_bought", 0), p["avg_cost"], p["last_price"])
                for s, p in acct.get("positions", {}).items()]
        pending = st.get("orders", [])[-30:]
    out.append(_table(["代码", "数量", "可卖", "成本", "现价"], rows))
    out.append("<h3>待执行 / 最近委托</h3>")
    out.append(_table(["代码", "方向", "数量", "价格", "状态", "备注"],
                      [(o["symbol"], o["direction"], o["quantity"], o.get("price") or "市价",
                        o["status"], o.get("reason") or o.get("tag", "")) for o in pending]))
    logs = st.get("logs", [])[-20:]
    if logs:
        out.append("<h3>日志</h3><pre>" + html.escape("\n".join(logs)) + "</pre>")
    return "\n".join(out)


def _table(head, rows) -> str:
    if not rows:
        return "<p class='muted'>无</p>"
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(f'{v:,.3f}' if isinstance(v, float) else str(v))}</td>" for v in r) + "</tr>"
        for r in rows)
    return f"<div class='wrap'><table><tr>{th}</tr>{body}</table></div>"


def render(root: Path) -> str:
    states = sorted(root.rglob("state/*.json"))
    reports = sorted(root.rglob("report.html"))
    parts = [_state_section(p) for p in states] or ["<p class='muted'>还没有模拟盘/实盘状态文件。运行 <code>qa paper -c 配置.toml</code> 后这里会出现账户。</p>"]
    links = "".join(f"<li><a href='/file/{html.escape(str(p.relative_to(root)))}'>"
                    f"{html.escape(str(p.relative_to(root)))}</a></li>" for p in reports) or "<li>无</li>"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60">
<title>量化看板</title><style>{CSS} a{{color:var(--a)}} pre{{white-space:pre-wrap;font-size:12px}}</style></head>
<body><h1>量化看板</h1>{''.join(parts)}<h2>回测报告</h2><ul>{links}</ul></body></html>"""


def serve(root: str | Path = ".", host: str = "127.0.0.1", port: int = 8050) -> None:
    root = Path(root).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/file/"):
                target = (root / self.path[len("/file/"):]).resolve()
                if root not in target.parents or not target.is_file() or target.suffix != ".html":
                    self.send_error(404)
                    return
                body = target.read_bytes()
            elif self.path in ("/", "/index.html"):
                body = render(root).encode("utf-8")
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"看板已启动: http://{host}:{port}  (Ctrl+C 退出)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
