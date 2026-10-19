import json
import shutil
import threading
import time
import tomllib
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from quantum_analyze.webui import tomlw
from quantum_analyze.webui.server import App, make_handler

ROOT = Path(__file__).resolve().parents[1]


def test_toml_writer_roundtrip():
    data = {"account": {"initial_cash": 200000, "fx_rates": {"HKD": 0.92}},
            "data": {"sources": {"CN": ["akshare", "baostock"]}, "cache_dir": "data_cache"},
            "rules": {"HK": {"lot_sizes": {"00700": 100}}},
            "strategy": {"name": "dual_ma", "params": {"fast": 5, "k": 2.0, "flag": True, "s": 'a"b\\c'}},
            "trading": {"gateway_options": {"path": "D:\\QMT\\userdata_mini"}}}
    assert tomllib.loads(tomlw.dumps(data, header="x")) == data


@pytest.fixture
def ws(tmp_path):
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    return tmp_path


def test_save_and_load_config(ws):
    app = App(ws)
    loaded = app.load_config("examples/quickstart.toml")
    cfg = loaded["config"]
    cfg["strategy"] = {"name": "momentum_rotation", "params": {"top_k": 2}}
    r = app.save_config({"name": "my test!", "config": cfg})
    assert r["path"] == "configs/mytest.toml" and "warning" not in r
    again = app.load_config(r["path"])["config"]
    assert again["strategy"]["params"]["top_k"] == 2
    bad = app.save_config({"name": "bad", "config": {**cfg, "strategy": {"name": "nope"}}})
    assert "warning" in bad
    with pytest.raises(ValueError):
        app.safe("../outside.toml")
    meta = app.meta()
    assert any(c["path"] == "configs/mytest.toml" for c in meta["configs"])
    assert {s["name"] for s in meta["strategies"]} >= {"momentum_rotation", "multi_factor"}


def test_backtest_job_produces_result(ws):
    app = App(ws)
    job = app.run({"action": "backtest", "config": "examples/quickstart.toml", "start": "2024-01-01"})
    deadline = time.time() + 120
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.2)
    while not job.result and job.status == "done" and time.time() < deadline:
        time.sleep(0.1)
    assert job.status == "done", "\n".join(job.lines[-20:])
    assert job.result["metrics"]["metrics"]["final_value"] > 0
    assert job.result["report"].endswith("report.html")


def test_live_send_requires_confirmation(ws):
    app = App(ws)
    with pytest.raises(ValueError):
        app.run({"action": "live_send", "config": "examples/quickstart.toml", "confirm": "yes"})
    with pytest.raises(ValueError):
        app.run({"action": "install", "module": "evil-package"})


def test_http_requires_token_and_local_host(ws):
    app = App(ws)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        page = opener.open(base + "/").read().decode("utf-8")
        assert app.token in page
        req = urllib.request.Request(base + "/api/run", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            opener.open(req)
        assert e.value.code == 403
        req = urllib.request.Request(base + "/api/config/save", method="POST",
                                     data=json.dumps({"name": "t", "text": "[account]\ninitial_cash = 1\n"}).encode(),
                                     headers={"X-QA-Token": app.token, "Content-Type": "application/json"})
        assert json.loads(opener.open(req).read())["path"] == "configs/t.toml"
        req = urllib.request.Request(base + "/api/meta", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError):
            opener.open(req)
    finally:
        httpd.shutdown()
