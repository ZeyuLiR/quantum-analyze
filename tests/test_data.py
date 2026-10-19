"""Offline tests for data sources and DataManager."""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import pytest

from quantum_analyze.data.base import DataSource, DataSourceError, DataSourceUnavailable
from quantum_analyze.data.manager import (DataManager, ValidationReport, build_source,
                                          cross_validate)
from quantum_analyze.data.sources import (AkShareSource, AlphaVantageSource, BaoStockSource,
                                          CSVSource, FutuSource, IBKRSource, QMTSource,
                                          SyntheticSource, TushareSource, YFinanceSource,
                                          futu_code, ib_duration, is_cn_index, yf_symbol)

START, END = "2023-01-01", "2023-12-31"


class FailingSource(DataSource):
    name = "failing"

    def __init__(self, exc: type[Exception] = DataSourceError):
        self.exc, self.calls = exc, 0

    def fetch(self, symbol, start, end, frequency="1d"):
        self.calls += 1
        raise self.exc("boom")


class CountingSource(DataSource):
    """Wraps SyntheticSource, cacheable, records each call's arguments."""
    name = "counting"

    def __init__(self, seed: int = 42, scale: float = 1.0):
        self.inner, self.scale, self.calls = SyntheticSource(seed), scale, []

    def fetch(self, symbol, start, end, frequency="1d"):
        self.calls.append((symbol, start, end))
        df = self.inner.fetch(symbol, start, end, frequency)
        df[["open", "high", "low", "close"]] *= self.scale
        return df


# ----------------------------------------------------------------- synthetic
def test_synthetic_deterministic_and_normalised():
    a = SyntheticSource(seed=7).load(["600519.SH", "AAPL.US"], START, END)
    b = SyntheticSource(seed=7).load(["600519.SH", "AAPL.US"], START, END)
    for sym in a:
        pd.testing.assert_frame_equal(a[sym], b[sym])
    df = a["600519.SH"]
    assert list(df.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert "pre_close" in df and "amount" in df
    assert df.index.is_monotonic_increasing and df.index.is_unique and df.index.tz is None
    assert (df.index.dayofweek < 5).all() and len(df) == len(pd.bdate_range(START, END))
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["volume"] > 0).all() and (df["volume"] % 100 == 0).all()
    assert not a["600519.SH"]["close"].equals(a["AAPL.US"]["close"])
    other = SyntheticSource(seed=8).fetch("600519.SH", START, END)
    assert not np.allclose(other["close"].to_numpy(), df["close"].to_numpy())


def test_synthetic_range_independent_and_price_limits():
    src = SyntheticSource(seed=1)
    full = src.fetch("300750.SZ", "2010-01-01", "2020-12-31")
    part = SyntheticSource(seed=1).fetch("300750.SZ", "2015-03-02", "2015-06-30")
    pd.testing.assert_frame_equal(part, full.loc["2015-03-02":"2015-06-30"])
    for sym, lim in (("600000.SH", 0.10), ("300750.SZ", 0.20)):
        df = src.load([sym], "2005-01-01", "2024-12-31")[sym]
        chg = (df["close"] / df["pre_close"] - 1).dropna().abs()
        assert chg.max() <= lim + 0.006  # rounding to 0.01
        assert ((df["high"] / df["pre_close"] - 1).dropna() <= lim + 0.006).all()
        assert df["close"].min() > 0


def test_synthetic_rejects_minute():
    with pytest.raises(DataSourceError):
        SyntheticSource().fetch("AAPL.US", START, END, "5m")


# ----------------------------------------------------------------- csv
def test_csv_roundtrip(tmp_path):
    df = SyntheticSource().load(["00700.HK"], START, END)["00700.HK"]
    src = CSVSource(tmp_path)
    src.save("00700.HK", df)
    back = src.load(["00700.HK"], START, END)["00700.HK"]
    pd.testing.assert_frame_equal(back, df, check_freq=False)
    sub = src.fetch("00700.HK", "2023-06-01", "2023-06-30")
    assert sub.index.min() >= pd.Timestamp("2023-06-01")
    assert sub.index.max() <= pd.Timestamp("2023-06-30")


def test_csv_chinese_headers(tmp_path):
    (tmp_path / "600519.SH.csv").write_text(
        "日期,开盘,最高,最低,收盘,成交量,成交额\n"
        "2024-01-03,1700,1720,1690,1710,30000,51000000\n"
        "2024-01-02,1680,1705,1675,1700,25000,42500000\n", encoding="utf-8")
    df = CSVSource(tmp_path, volume_multiplier=100).load(["600519"])["600519.SH"]
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df.loc["2024-01-03", "close"] == 1710 and df.loc["2024-01-03", "pre_close"] == 1700
    assert df.loc["2024-01-02", "volume"] == 2_500_000 and "amount" in df


def test_csv_trade_date_ints_and_missing(tmp_path):
    pd.DataFrame({"trade_date": [20240102, 20240103], "open": [1, 2], "high": [2, 3],
                  "low": [0.5, 1.5], "close": [1.5, 2.5], "vol": [10, 20]}).to_csv(
        tmp_path / "AAPL.US.csv", index=False)
    df = CSVSource(tmp_path).load(["AAPL.US"])["AAPL.US"]
    assert df.index[0] == pd.Timestamp("2024-01-02") and df["volume"].tolist() == [10, 20]
    with pytest.raises(DataSourceError):
        CSVSource(tmp_path).fetch("MSFT.US", None, None)


# ----------------------------------------------------------------- manager
def test_fallback_and_provenance(caplog):
    bad, unavailable = FailingSource(), FailingSource(DataSourceUnavailable)
    dm = DataManager({"CN": [unavailable, bad, SyntheticSource()], "US": [SyntheticSource()]})
    with caplog.at_level(logging.INFO):
        out = dm.load(["600519.SH", "AAPL"], START, END)
    assert set(out) == {"600519.SH", "AAPL.US"}
    assert dm.provenance == {"600519.SH": "synthetic", "AAPL.US": "synthetic"}
    assert bad.calls == 1 and unavailable.calls == 1
    assert set(dm.errors["600519.SH"]) == {"failing"}
    pd.testing.assert_frame_equal(out["600519.SH"],
                                  SyntheticSource().load(["600519.SH"], START, END)["600519.SH"])


def test_all_sources_fail():
    dm = DataManager({"CN": [FailingSource()]})
    with pytest.raises(DataSourceError, match="所有数据源均失败"):
        dm.load(["600519.SH"], START, END)
    assert dm.load(["600519.SH"], START, END, errors="skip") == {}
    with pytest.raises(DataSourceUnavailable):
        dm.load(["AAPL.US"], START, END)  # no US sources configured


def test_cache_reuse_incremental_and_refresh(tmp_path):
    src = CountingSource()
    dm = DataManager({"CN": [src]}, cache_dir=tmp_path, cache_format="csv")
    first = dm.load(["600519.SH"], START, "2023-06-30")["600519.SH"]
    second = DataManager({"CN": [src]}, cache_dir=tmp_path).load(
        ["600519.SH"], "2023-03-01", "2023-06-30")["600519.SH"]
    assert len(src.calls) == 1
    pd.testing.assert_frame_equal(second, first.loc["2023-03-01":], check_freq=False)
    assert list(tmp_path.glob("counting/1d/600519.SH.*"))
    # extending the end only fetches the tail
    ext = dm.load(["600519.SH"], START, END)["600519.SH"]
    assert len(src.calls) == 2 and src.calls[-1][1] >= "2023-06-20"
    ref = SyntheticSource().fetch("600519.SH", START, END)
    np.testing.assert_allclose(ext["close"].to_numpy(), ref["close"].to_numpy())
    assert ext["pre_close"].isna().sum() == 1
    # earlier start needs the head -> full refetch; then refresh bypasses the cache
    dm.load(["600519.SH"], "2022-06-01", END)
    assert len(src.calls) == 3 and src.calls[-1][1] == "2022-06-01"
    dm.load(["600519.SH"], START, END)
    assert len(src.calls) == 3
    dm.load(["600519.SH"], START, END, refresh=True)
    assert len(src.calls) == 4


def test_cache_detects_readjustment(tmp_path):
    src = CountingSource()
    dm = DataManager({"CN": [src]}, cache_dir=tmp_path, cache_format="csv")
    dm.load(["600519.SH"], START, "2023-06-30")
    src.scale = 0.9  # simulate an ex-dividend re-adjustment of the whole history
    out = dm.load(["600519.SH"], START, END)["600519.SH"]
    assert src.calls[-1][1] == START  # tail mismatch -> full refetch
    ref = SyntheticSource().fetch("600519.SH", START, END)
    np.testing.assert_allclose(out["close"].to_numpy(), ref["close"].to_numpy() * 0.9)


def test_cross_validation(caplog):
    same = cross_validate("AAPL.US", [SyntheticSource(), CountingSource()], START, END)
    assert isinstance(same, ValidationReport) and same.ok
    assert same.overlap == len(pd.bdate_range(START, END)) and same.max_deviation == 0
    off = DataManager({"US": [SyntheticSource(), CountingSource(scale=1.05)]})
    rep = off.cross_validate("AAPL.US", start=START, end=END, tolerance=0.02)
    assert not rep.ok and rep.mismatch_ratio == 1.0 and rep.max_deviation == pytest.approx(0.05 / 1.05)
    dm = DataManager({"US": [SyntheticSource(), CountingSource(scale=1.05)]}, validate=True)
    with caplog.at_level(logging.WARNING):
        dm.load(["AAPL.US"], START, END)
    assert not dm.validation["AAPL.US"].ok
    assert any("交叉验证不一致" in r.getMessage() for r in caplog.records)


def test_from_config_and_build_source(tmp_path):
    dm = DataManager.from_config({"sources": "synthetic", "options": {"synthetic": {"seed": 3}}})
    assert all(s[0].name == "synthetic" and s[0].seed == 3 for s in dm.sources.values())
    assert dm.sources["CN"][0] is dm.sources["US"][0]
    cfg = {"cache_dir": str(tmp_path / "cache"), "validate": False,
           "sources": {"CN": ["akshare", "baostock", "tushare", "qmt"],
                       "HK": ["futu", "yfinance", "ibkr"],
                       "US": ["yfinance", "ibkr", "alphavantage", "futu"]},
           "options": {"tushare": {"token": "x"}, "futu": {"host": "127.0.0.1", "port": 11111},
                       "ibkr": {"host": "127.0.0.1", "port": 7497, "client_id": 17},
                       "csv": {"path": "data"}}}
    dm = DataManager.from_config(cfg)
    assert [s.name for s in dm.sources["CN"]] == ["akshare", "baostock", "tushare", "qmt"]
    assert dm.sources["HK"][0] is dm.sources["US"][3] and dm.sources["HK"][0].port == 11111
    assert dm.sources["HK"][2].client_id == 17 and dm.sources["CN"][2].token == "x"
    assert dm.cache is not None
    assert isinstance(build_source("csv", path=str(tmp_path)), CSVSource)
    with pytest.raises(ValueError):
        build_source("bloomberg")
    with pytest.raises(ValueError):
        build_source("futu", bogus=1)


def test_csv_via_config(tmp_path):
    SyntheticSource().load(["AAPL.US"], START, END)  # warm
    CSVSource(tmp_path).save("AAPL.US", SyntheticSource().fetch("AAPL.US", START, END))
    dm = DataManager.from_config({"sources": "csv", "options": {"csv": {"path": str(tmp_path)}}})
    assert len(dm.load(["AAPL.US"], START, END)["AAPL.US"]) == len(pd.bdate_range(START, END))
    assert dm.provenance["AAPL.US"] == "csv"


# ----------------------------------------------------------------- optional sources
OPTIONAL = [
    (AkShareSource(), "600519.SH", ["akshare"]),
    (BaoStockSource(), "600519.SH", ["baostock"]),
    (TushareSource(token="t"), "600519.SH", ["tushare"]),
    (QMTSource(), "600519.SH", ["xtquant", "xtquant.xtdata"]),
    (FutuSource(), "00700.HK", ["futu"]),
    (YFinanceSource(), "AAPL.US", ["yfinance"]),
    (IBKRSource(), "AAPL.US", ["ib_insync", "ib_async"]),
]


@pytest.mark.parametrize("src,symbol,modules", OPTIONAL, ids=[o[0].name for o in OPTIONAL])
def test_missing_package_raises_unavailable(monkeypatch, src, symbol, modules):
    for m in modules:
        monkeypatch.setitem(sys.modules, m, None)
    with pytest.raises(DataSourceUnavailable, match="pip install"):
        src.fetch(symbol, START, END)


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(DataSourceUnavailable, match="TUSHARE_TOKEN"):
        TushareSource().fetch("600519.SH", START, END)
    with pytest.raises(DataSourceUnavailable, match="ALPHAVANTAGE_API_KEY"):
        AlphaVantageSource().fetch("AAPL.US", START, END)


def test_manager_falls_back_past_missing_packages(monkeypatch):
    for m in ("akshare", "baostock", "yfinance"):
        monkeypatch.setitem(sys.modules, m, None)
    dm = DataManager.from_config({"sources": {"CN": ["akshare", "baostock", "synthetic"],
                                              "US": ["yfinance", "synthetic"]}})
    dm.load(["600519.SH", "AAPL.US"], START, END)
    assert dm.provenance == {"600519.SH": "synthetic", "AAPL.US": "synthetic"}


# ----------------------------------------------------------------- symbol mapping
def test_symbol_conversions():
    assert yf_symbol("00700.HK") == "0700.HK" and yf_symbol("09988.HK") == "9988.HK"
    assert yf_symbol("600519.SH") == "600519.SS" and yf_symbol("BRK.B.US") == "BRK-B"
    assert futu_code("700") == "HK.00700" and futu_code("AAPL") == "US.AAPL"
    assert futu_code("000001.SZ") == "SZ.000001"
    assert is_cn_index("000300.SH") and is_cn_index("399006.SZ")
    assert not is_cn_index("000001.SZ") and is_cn_index("H30269.CSI", {"H30269.CSI"})
    assert ib_duration("2024-01-01", pd.Timestamp("2024-01-31")) == "31 D"
    assert ib_duration("2020-01-01", pd.Timestamp("2024-01-31")) == "5 Y"
    assert not BaoStockSource().supports("430047.BJ")
    assert not AkShareSource().supports("000300.SH", "5m") and AkShareSource().supports("000300.SH")
    assert not AlphaVantageSource().supports("00700.HK")
