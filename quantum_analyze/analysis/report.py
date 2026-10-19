"""Self-contained HTML backtest report (inline SVG, no plotting dependency)."""
from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .metrics import drawdown, format_metrics, to_daily

if TYPE_CHECKING:
    from ..engine import BacktestResult

CSS = """
:root{--bg:#fff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--a:#2563eb;--b:#9ca3af;--neg:#dc2626;--card:#f6f8fa}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--a:#60a5fa;--b:#6b7280;--neg:#f87171;--card:#161b22}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:20px}h2{font-size:16px;margin-top:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px}
.kpi b{display:block;font-size:16px}.kpi span{color:var(--muted);font-size:12px}
svg{width:100%;height:auto;display:block}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}
th:first-child,td:first-child{text-align:left}.wrap{overflow-x:auto}.muted{color:var(--muted)}
"""


def _path(series: pd.Series, w: int, h: int, lo: float, hi: float) -> str:
    n = len(series)
    if n < 2 or hi <= lo:
        return ""
    pts = []
    for i, v in enumerate(series.to_numpy()):
        x = i / (n - 1) * w
        y = h - (v - lo) / (hi - lo) * h
        pts.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(pts)


def line_chart(series: dict[str, pd.Series], colors: dict[str, str], w=1000, h=260,
               fill_zero=False) -> str:
    vals = pd.concat(series.values())
    lo, hi = float(vals.min()), float(vals.max())
    if fill_zero:
        hi = max(hi, 0.0)
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    body = []
    for name, s in series.items():
        d = _path(s, w, h, lo, hi)
        if fill_zero:
            zero_y = h - (0 - lo) / (hi - lo) * h
            body.append(f'<path d="{d} L{w},{zero_y:.1f} L0,{zero_y:.1f} Z" fill="{colors[name]}" opacity=".25"/>')
        body.append(f'<path d="{d}" fill="none" stroke="{colors[name]}" stroke-width="1.6"/>')
    first, last = next(iter(series.values())).index[[0, -1]]
    legend = " ".join(f'<tspan fill="{colors[n]}">■</tspan> {html.escape(n)}' for n in series)
    return (f'<svg viewBox="0 -18 {w} {h + 36}" role="img">'
            f'<text x="0" y="-4" font-size="12" fill="currentColor">{legend}</text>'
            f'<text x="{w}" y="-4" font-size="12" text-anchor="end" fill="currentColor">{hi:.3g} / {lo:.3g}</text>'
            + "".join(body) +
            f'<text x="0" y="{h + 14}" font-size="11" fill="currentColor">{first.date()}</text>'
            f'<text x="{w}" y="{h + 14}" font-size="11" text-anchor="end" fill="currentColor">{last.date()}</text></svg>')


def _table(df: pd.DataFrame, limit: int = 300) -> str:
    if df is None or df.empty:
        return '<p class="muted">无</p>'
    df = df.tail(limit)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            s = f"{v:,.2f}" if isinstance(v, float) else str(v)
            cells.append(f"<td>{html.escape(s)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="wrap"><table><tr>{head}</tr>{"".join(rows)}</table></div>'


def write_html_report(result: "BacktestResult", path, title: Optional[str] = None) -> Path:
    eq = to_daily(result.equity["total"])
    nav = eq / eq.iloc[0]
    series = {"策略净值": nav}
    colors = {"策略净值": "var(--a)", "基准": "var(--b)"}
    if result.benchmark is not None and len(result.benchmark.dropna()) > 1:
        b = to_daily(result.benchmark).reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            series["基准"] = b / b.iloc[0]
    dd = drawdown(eq)
    kpis = "".join(f'<div class="kpi"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>'
                   for k, v in format_metrics(result.metrics))
    monthly = eq.resample("ME").last().pct_change().dropna()
    mt = pd.DataFrame({"年": monthly.index.year, "月": monthly.index.month, "r": monthly.values})
    mtable = mt.pivot(index="年", columns="月", values="r") if len(mt) else pd.DataFrame()
    if len(mtable):
        mtable = mtable.map(lambda v: "" if pd.isna(v) else f"{v:+.1%}").reset_index()
    trades = result.trades.drop(columns=[c for c in ("fx", "trade_id", "order_id") if c in result.trades])
    events = "".join(f"<li>{html.escape(e)}</li>" for e in result.risk_events[-100:]) or "<li>无</li>"
    doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title or '回测报告')}</title>
<style>{CSS}</style></head><body>
<h1>{html.escape(title or '回测报告')}：{html.escape(result.strategy)}</h1>
<p class="muted">参数 {html.escape(str(result.params))}</p>
<div class="grid">{kpis}</div>
<h2>净值曲线</h2>{line_chart(series, colors)}
<h2>回撤</h2>{line_chart({'回撤': dd}, {'回撤': 'var(--neg)'}, h=140, fill_zero=True)}
<h2>月度收益</h2>{_table(mtable)}
<h2>风控事件</h2><ul>{events}</ul>
<h2>成交记录（最近 300 笔）</h2>{_table(trades)}
</body></html>"""
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
