"""Concrete data sources.

Online adapters (AkShare, BaoStock, Tushare, miniQMT, Futu, yfinance, IBKR,
Alpha Vantage) import their SDK lazily inside ``fetch`` and raise
:class:`DataSourceUnavailable` with an install hint when it is missing or a
required token is absent. :class:`CSVSource` and :class:`SyntheticSource` work
fully offline and are used by tests and demos.

Every ``fetch`` returns bars in the convention described in :mod:`.base`:
forward-adjusted (前复权) prices, volume in shares, amount in the market's
currency, clipped to ``[start, end]``.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
import math
import os
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from ..core.rules import a_share_spec, market_of, normalize_symbol
from .base import DataSource, DataSourceError, DataSourceUnavailable, clip_dates

log = logging.getLogger(__name__)

MINUTE_FREQS = ("1m", "5m", "15m", "30m", "60m")
ALL_FREQS = ("1d",) + MINUTE_FREQS
ETF_PREFIXES = ("51", "56", "58", "15", "16")

# header aliases (lower-cased English + Chinese) -> canonical column
HEADER_ALIASES = {
    "date": "date", "datetime": "date", "time": "date", "trade_date": "date", "trade_time": "date",
    "timestamp": "date", "time_key": "date", "日期": "date", "时间": "date", "交易日期": "date",
    "open": "open", "开盘": "open", "开盘价": "open",
    "high": "high", "最高": "high", "最高价": "high",
    "low": "low", "最低": "low", "最低价": "low",
    "close": "close", "收盘": "close", "收盘价": "close",
    "volume": "volume", "vol": "volume", "成交量": "volume",
    "amount": "amount", "turnover": "amount", "成交额": "amount", "成交金额": "amount",
    "pre_close": "pre_close", "preclose": "pre_close", "昨收": "pre_close", "前收盘": "pre_close",
}


# ----------------------------------------------------------------- helpers
def _require(module: str, pip: str) -> Any:
    """Import an optional dependency or raise DataSourceUnavailable with a hint."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise DataSourceUnavailable(f"未安装 {module}，请先运行: pip install {pip}") from e


@contextlib.contextmanager
def _wrap(source: str, symbol: str) -> Iterator[None]:
    """Turn any third-party exception into a DataSourceError."""
    try:
        yield
    except DataSourceError:
        raise
    except Exception as e:  # noqa: BLE001 - SDKs raise anything
        raise DataSourceError(f"{source} 获取 {symbol} 失败: {type(e).__name__}: {e}") from e


def split_symbol(symbol: str) -> tuple[str, str]:
    """``600519.SH`` -> (``600519``, ``SH``); ``BRK.B.US`` -> (``BRK.B``, ``US``)."""
    code, mkt = normalize_symbol(symbol).rsplit(".", 1)
    return code, mkt


def _fmt(d: Optional[str], fmt: str, default: Any = None) -> Any:
    return pd.Timestamp(d).strftime(fmt) if d else default


def is_etf(code: str) -> bool:
    return len(code) == 6 and code.isdigit() and code.startswith(ETF_PREFIXES)


def is_cn_index(symbol: str, indices: Iterable[str] = ()) -> bool:
    """Indices: explicitly listed, ``000xxx.SH`` (e.g. 000300.SH = 沪深300) or ``399xxx.SZ``."""
    s = normalize_symbol(symbol)
    if s in indices:
        return True
    code, mkt = s.rsplit(".", 1)
    return (mkt == "SH" and code.startswith("000")) or (mkt == "SZ" and code.startswith("399"))


def to_bars(df: Optional[pd.DataFrame], rename: Optional[dict[str, str]] = None,
            volume_mult: float = 1.0, amount_mult: float = 1.0,
            date_format: Optional[str] = None) -> pd.DataFrame:
    """Map a raw SDK frame onto open/high/low/close/volume[/amount] indexed by datetime."""
    if df is None or len(df) == 0:
        raise DataSourceError("数据源返回空数据")
    df = df.rename(columns=rename) if rename else df
    if "date" in df.columns:
        raw = df["date"].astype(str) if date_format else df["date"]
        idx = pd.DatetimeIndex(pd.to_datetime(raw, format=date_format))
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:  # keep exchange wall-clock time, drop tz
        idx = idx.tz_localize(None)
    cols = [c for c in ("open", "high", "low", "close", "volume", "amount", "pre_close") if c in df.columns]
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols},
                       index=idx)
    out["volume"] = out["volume"] * volume_mult
    if "amount" in out:
        out["amount"] = out["amount"] * amount_mult
    out.index.name = "datetime"
    return out


def parse_bar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Parse a user-supplied table (English or Chinese headers) into bars."""
    rename: dict[str, str] = {}
    for c in df.columns:
        key = HEADER_ALIASES.get(str(c).strip().lower(), HEADER_ALIASES.get(str(c).strip()))
        if key and key not in rename.values():
            rename[c] = key
    df = df.rename(columns=rename)
    if "date" not in df.columns:  # e.g. a CSV saved with an unnamed index column
        df = df.rename(columns={df.columns[0]: "date"})
    if pd.api.types.is_integer_dtype(df["date"]):  # 20240102
        df["date"] = df["date"].astype(str)
    return to_bars(df)


# ----------------------------------------------------------------- AkShare
class AkShareSource(DataSource):
    """AkShare (东方财富 endpoints). CN stocks/ETFs/indices, HK and US daily; CN minute bars."""

    name = "akshare"
    markets = ("CN", "HK", "US")
    frequencies = ALL_FREQS

    def __init__(self, adjust: str = "qfq", indices: Optional[Iterable[str]] = None):
        self.adjust = adjust
        self.indices = {normalize_symbol(s) for s in (indices or ())}
        self._us_prefix: dict[str, str] = {}

    def supports(self, symbol: str, frequency: str = "1d") -> bool:
        if frequency == "1d":
            return market_of(symbol) in self.markets
        return (frequency in MINUTE_FREQS and market_of(symbol) == "CN"
                and not is_cn_index(symbol, self.indices))

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        ak = _require("akshare", "akshare")
        code, mkt = split_symbol(symbol)
        s, e = _fmt(start, "%Y%m%d", "19700101"), _fmt(end, "%Y%m%d", "20500101")
        with _wrap(self.name, symbol):
            if frequency != "1d":
                fn = ak.fund_etf_hist_min_em if is_etf(code) else ak.stock_zh_a_hist_min_em
                p = frequency[:-1]
                df = fn(symbol=code, period=p, adjust="" if p == "1" else self.adjust,
                        start_date=_fmt(start, "%Y-%m-%d 09:00:00", "1979-09-01 09:00:00"),
                        end_date=_fmt(end, "%Y-%m-%d 15:30:00", "2222-01-01 15:30:00"))
                bars = to_bars(df, HEADER_ALIASES, volume_mult=100)  # 手 -> 股
            elif mkt == "US":
                bars = to_bars(self._us_daily(ak, code, s, e), HEADER_ALIASES)
            elif mkt == "HK":
                df = ak.stock_hk_hist(symbol=code.zfill(5), period="daily", start_date=s,
                                      end_date=e, adjust=self.adjust)
                bars = to_bars(df, HEADER_ALIASES)
            elif is_cn_index(symbol, self.indices):
                sym = f"{mkt.lower()}{code}"
                try:
                    df = ak.stock_zh_index_daily_em(symbol=sym, start_date=s, end_date=e)
                except TypeError:  # older akshare without date arguments
                    df = ak.stock_zh_index_daily_em(symbol=sym)
                bars = to_bars(df, volume_mult=100)
            else:
                fn = ak.fund_etf_hist_em if is_etf(code) else ak.stock_zh_a_hist
                df = fn(symbol=code, period="daily", start_date=s, end_date=e, adjust=self.adjust)
                bars = to_bars(df, HEADER_ALIASES, volume_mult=100)
        return clip_dates(bars, start, end)

    def _us_daily(self, ak: Any, code: str, s: str, e: str) -> pd.DataFrame:
        """East Money US codes need a market prefix: 105 NASDAQ, 106 NYSE, 107 AMEX."""
        ticker = code.replace(".", "_")
        prefixes = [self._us_prefix[code]] if code in self._us_prefix else ["105", "106", "107"]
        for p in prefixes:
            try:
                df = ak.stock_us_hist(symbol=f"{p}.{ticker}", period="daily", start_date=s,
                                      end_date=e, adjust=self.adjust)
            except Exception:  # noqa: BLE001 - wrong prefix raises inside akshare
                continue
            if df is not None and len(df):
                self._us_prefix[code] = p
                return df
        raise DataSourceError(f"AkShare 找不到美股 {code}（已尝试 105/106/107 前缀）")


# ----------------------------------------------------------------- BaoStock
class BaoStockSource(DataSource):
    """BaoStock (free, login required). SH/SZ stocks and indices; 5/15/30/60 minute bars."""

    name = "baostock"
    markets = ("CN",)
    frequencies = ("1d", "5m", "15m", "30m", "60m")

    def __init__(self, adjustflag: str = "2", indices: Optional[Iterable[str]] = None):
        self.adjustflag = adjustflag  # 1 后复权, 2 前复权, 3 不复权
        self.indices = {normalize_symbol(s) for s in (indices or ())}

    def supports(self, symbol: str, frequency: str = "1d") -> bool:
        if not super().supports(symbol, frequency) or symbol.upper().endswith(".BJ"):
            return False
        return frequency == "1d" or not is_cn_index(symbol, self.indices)

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        bs = _require("baostock", "baostock")
        code, mkt = split_symbol(symbol)
        if mkt not in ("SH", "SZ"):
            raise DataSourceError(f"BaoStock 不支持 {symbol}（仅沪深）")
        daily, idx = frequency == "1d", is_cn_index(symbol, self.indices)
        fields = ("date,open,high,low,close,volume,amount" + ("" if idx else ",tradestatus")
                  if daily else "date,time,open,high,low,close,volume,amount")
        with contextlib.redirect_stdout(io.StringIO()):  # baostock prints login banners
            lg = bs.login()
        if lg.error_code != "0":
            raise DataSourceUnavailable(f"BaoStock 登录失败: {lg.error_msg}")
        try:
            with _wrap(self.name, symbol):
                rs = bs.query_history_k_data_plus(
                    f"{mkt.lower()}.{code}", fields,
                    start_date=_fmt(start, "%Y-%m-%d", "1990-12-19"),
                    end_date=_fmt(end, "%Y-%m-%d", None),
                    frequency="d" if daily else frequency[:-1],
                    adjustflag="3" if idx else self.adjustflag)
                if rs.error_code != "0":
                    raise DataSourceError(f"BaoStock 查询 {symbol} 失败: {rs.error_msg}")
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                df = pd.DataFrame(rows, columns=rs.fields)
        finally:
            with contextlib.redirect_stdout(io.StringIO()):
                bs.logout()
        if "tradestatus" in df:
            df = df[df["tradestatus"] == "1"]  # drop suspended days
        if daily:
            bars = to_bars(df)
        else:  # time looks like 20240102093500000
            bars = to_bars(df.assign(date=df["time"].str[:14]), date_format="%Y%m%d%H%M%S")
        return clip_dates(bars, start, end)


# ----------------------------------------------------------------- Tushare
class TushareSource(DataSource):
    """Tushare Pro ``pro_bar`` (token from constructor or env ``TUSHARE_TOKEN``)."""

    name = "tushare"
    markets = ("CN",)
    frequencies = ALL_FREQS

    def __init__(self, token: Optional[str] = None, adj: str = "qfq",
                 indices: Optional[Iterable[str]] = None):
        self.token = token
        self.adj = adj
        self.indices = {normalize_symbol(s) for s in (indices or ())}

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        token = self.token or os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise DataSourceUnavailable("缺少 Tushare token：请设置环境变量 TUSHARE_TOKEN "
                                        "或在配置 options.tushare.token 中提供")
        ts = _require("tushare", "tushare")
        code, _ = split_symbol(symbol)
        asset = "I" if is_cn_index(symbol, self.indices) else "FD" if is_etf(code) else "E"
        daily = frequency == "1d"
        with _wrap(self.name, symbol):
            df = ts.pro_bar(
                ts_code=normalize_symbol(symbol), api=ts.pro_api(token), asset=asset,
                adj=self.adj if (daily and asset != "I") else None,
                freq="D" if daily else f"{frequency[:-1]}min",
                start_date=_fmt(start, "%Y%m%d" if daily else "%Y-%m-%d 09:00:00", "19900101"),
                end_date=_fmt(end, "%Y%m%d" if daily else "%Y-%m-%d 15:30:00", None))
            if df is None:
                raise DataSourceError(f"Tushare 未返回 {symbol} 数据（积分/权限不足或代码错误）")
            if daily:  # vol 手, amount 千元
                bars = to_bars(df.rename(columns={"trade_date": "date", "vol": "volume"}),
                               volume_mult=100, amount_mult=1000, date_format="%Y%m%d")
            else:      # stk_mins: vol 股, amount 元
                bars = to_bars(df.rename(columns={"trade_time": "date", "vol": "volume"}))
        return clip_dates(bars, start, end)


# ----------------------------------------------------------------- miniQMT
class QMTSource(DataSource):
    """miniQMT local data via ``xtquant.xtdata`` (the QMT client must be running)."""

    name = "qmt"
    markets = ("CN",)
    frequencies = ALL_FREQS
    PERIODS = {"1d": "1d", "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "1h"}

    def __init__(self, dividend_type: str = "front", download: bool = True):
        self.dividend_type = dividend_type  # front 前复权, back 后复权, none
        self.download = download

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        xtdata = _require("xtquant.xtdata", "xtquant  （或使用 miniQMT 自带的 xtquant）")
        sym, period = normalize_symbol(symbol), self.PERIODS[frequency]
        st = _fmt(start, "%Y%m%d", "")
        et = _fmt(end, "%Y%m%d" if frequency == "1d" else "%Y%m%d235959", "")
        with _wrap(self.name, symbol):
            if self.download:
                xtdata.download_history_data(sym, period=period, start_time=st, end_time=et)
            data = xtdata.get_market_data_ex(
                ["time", "open", "high", "low", "close", "volume", "amount"], [sym],
                period=period, start_time=st, end_time=et, count=-1,
                dividend_type=self.dividend_type, fill_data=False)
            df = data.get(sym) if data else None
            if df is None or len(df) == 0:
                raise DataSourceError(f"miniQMT 没有 {sym} 的 {period} 数据")
            t = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
            bars = to_bars(df.assign(date=t.dt.tz_localize(None).to_numpy()), volume_mult=100)
        return clip_dates(bars, start, end)


# ----------------------------------------------------------------- Futu
def futu_code(symbol: str) -> str:
    code, mkt = split_symbol(symbol)
    if mkt == "HK":
        return f"HK.{code.zfill(5)}"
    if mkt in ("SH", "SZ"):
        return f"{mkt}.{code}"
    if mkt == "US":
        return f"US.{code}"
    raise DataSourceError(f"富途不支持 {symbol}")


class FutuSource(DataSource):
    """Futu OpenAPI via a running OpenD gateway (``pip install futu-api``)."""

    name = "futu"
    markets = ("CN", "HK", "US")
    frequencies = ALL_FREQS
    KTYPES = {"1d": "K_DAY", "1m": "K_1M", "5m": "K_5M", "15m": "K_15M",
              "30m": "K_30M", "60m": "K_60M"}

    def __init__(self, host: str = "127.0.0.1", port: int = 11111, autype: str = "QFQ",
                 max_count: int = 1000):
        self.host, self.port, self.autype, self.max_count = host, int(port), autype, max_count

    def supports(self, symbol: str, frequency: str = "1d") -> bool:
        return super().supports(symbol, frequency) and not symbol.upper().endswith(".BJ")

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        futu = _require("futu", "futu-api")
        code = futu_code(symbol)
        frames: list[pd.DataFrame] = []
        with _wrap(self.name, symbol):
            ctx = futu.OpenQuoteContext(host=self.host, port=self.port)
            try:
                key = None
                while True:
                    ret, data, key = ctx.request_history_kline(
                        code, start=_fmt(start, "%Y-%m-%d", "1990-01-01"),
                        end=_fmt(end, "%Y-%m-%d", None),
                        ktype=getattr(futu.KLType, self.KTYPES[frequency]),
                        autype=getattr(futu.AuType, self.autype),
                        max_count=self.max_count, page_req_key=key)
                    if ret != futu.RET_OK:
                        raise DataSourceError(f"富途获取 {code} 失败: {data}")
                    frames.append(data)
                    if key is None:
                        break
            finally:
                ctx.close()
            df = pd.concat(frames, ignore_index=True)
            bars = to_bars(df.rename(columns={"time_key": "date", "turnover": "amount"}))  # volume 股
        return clip_dates(bars, start, end)


# ----------------------------------------------------------------- yfinance
def yf_symbol(symbol: str) -> str:
    """``00700.HK`` -> ``0700.HK``; ``600519.SH`` -> ``600519.SS``; ``BRK.B.US`` -> ``BRK-B``."""
    code, mkt = split_symbol(symbol)
    if mkt == "HK":
        return f"{(code.lstrip('0') or '0').zfill(4)}.HK"
    if mkt in ("SH", "SS"):
        return f"{code}.SS"
    if mkt == "SZ":
        return f"{code}.SZ"
    if mkt == "BJ":
        raise DataSourceError(f"yfinance 不支持北交所 {symbol}")
    return code.replace(".", "-")


class YFinanceSource(DataSource):
    """Yahoo Finance via ``yfinance`` (auto-adjusted prices)."""

    name = "yfinance"
    markets = ("CN", "HK", "US")
    frequencies = ALL_FREQS

    def supports(self, symbol: str, frequency: str = "1d") -> bool:
        return super().supports(symbol, frequency) and not symbol.upper().endswith(".BJ")

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        yf = _require("yfinance", "yfinance")
        ysym = yf_symbol(symbol)
        # yfinance's end is exclusive
        end_x = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else None
        with _wrap(self.name, symbol):
            kw: dict[str, Any] = dict(interval=frequency, auto_adjust=True, actions=False)
            if start:
                kw.update(start=pd.Timestamp(start).strftime("%Y-%m-%d"), end=end_x)
            else:
                kw.update(period="max" if frequency == "1d" else "7d")
                if end_x:
                    kw["end"] = end_x
            df = yf.Ticker(ysym).history(**kw)
            if df is None or df.empty:
                raise DataSourceError(f"yfinance 没有 {ysym} 的数据")
            bars = to_bars(df.rename(columns=str.lower))
        return clip_dates(bars, start, end)


# ----------------------------------------------------------------- IBKR
def _import_ib() -> Any:
    for mod in ("ib_insync", "ib_async"):
        try:
            return importlib.import_module(mod)
        except ImportError:
            continue
    raise DataSourceUnavailable("未安装 ib_insync / ib_async，请先运行: pip install ib_async")


def ib_duration(start: Optional[str], ref_end: pd.Timestamp) -> str:
    """IB durationStr covering [start, ref_end]: 'N D' up to a year, else 'N Y'."""
    if not start:
        return "20 Y"
    days = max((ref_end.normalize() - pd.Timestamp(start).normalize()).days + 1, 1)
    return f"{days} D" if days <= 365 else f"{math.ceil(days / 365)} Y"


class IBKRSource(DataSource):
    """Interactive Brokers TWS / IB Gateway via ib_insync (or its fork ib_async)."""

    name = "ibkr"
    markets = ("HK", "US")
    frequencies = ALL_FREQS
    BAR_SIZES = {"1d": "1 day", "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
                 "30m": "30 mins", "60m": "1 hour"}
    TZ = {"HK": "Asia/Hong_Kong", "US": "America/New_York"}

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17,
                 timeout: float = 10.0, use_rth: bool = True):
        self.host, self.port, self.client_id = host, int(port), int(client_id)
        self.timeout, self.use_rth = timeout, use_rth

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        ib_mod = _import_ib()
        code, mkt = split_symbol(symbol)
        contract = (ib_mod.Stock(code.lstrip("0") or "0", "SEHK", "HKD") if mkt == "HK"
                    else ib_mod.Stock(code.replace(".", " "), "SMART", "USD"))
        today = pd.Timestamp.today().normalize()
        daily = frequency == "1d"
        # ADJUSTED_LAST only works with an empty endDateTime (= now)
        hist_end = not daily and end and pd.Timestamp(end).normalize() < today
        end_dt = pd.Timestamp(end).strftime("%Y%m%d-23:59:59") if hist_end else ""
        ib = ib_mod.IB()
        try:
            ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout,
                       readonly=True)
        except Exception as e:  # noqa: BLE001
            raise DataSourceUnavailable(f"无法连接 TWS/IB Gateway {self.host}:{self.port}: {e}") from e
        try:
            with _wrap(self.name, symbol):
                ib.qualifyContracts(contract)
                bars = ib.reqHistoricalData(
                    contract, endDateTime=end_dt,
                    durationStr=ib_duration(start, pd.Timestamp(end) if hist_end else today),
                    barSizeSetting=self.BAR_SIZES[frequency],
                    whatToShow="ADJUSTED_LAST" if daily else "TRADES",
                    useRTH=self.use_rth, formatDate=1)
                df = ib_mod.util.df(bars)
                if df is None or df.empty:
                    raise DataSourceError(f"IBKR 没有 {symbol} 的数据")
                idx = pd.to_datetime(df["date"])
                if getattr(idx.dt, "tz", None) is not None:
                    idx = idx.dt.tz_convert(self.TZ[mkt]).dt.tz_localize(None)
                out = to_bars(df.assign(date=idx.to_numpy()))
        finally:
            ib.disconnect()
        return clip_dates(out, start, end)


# ----------------------------------------------------------------- Alpha Vantage
class AlphaVantageSource(DataSource):
    """Alpha Vantage REST API (US daily only); key from constructor or ``ALPHAVANTAGE_API_KEY``."""

    name = "alphavantage"
    markets = ("US",)
    frequencies = ("1d",)
    URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, adjusted: bool = True, timeout: float = 30.0):
        self.api_key, self.adjusted, self.timeout = api_key, adjusted, timeout

    def _get(self, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310 - fixed https host
            return json.loads(resp.read().decode("utf-8"))

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        key = self.api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not key:
            raise DataSourceUnavailable("缺少 Alpha Vantage API key：请设置环境变量 ALPHAVANTAGE_API_KEY")
        code, _ = split_symbol(symbol)
        attempts = [("TIME_SERIES_DAILY", "full"), ("TIME_SERIES_DAILY", "compact")]
        if self.adjusted:
            attempts.insert(0, ("TIME_SERIES_DAILY_ADJUSTED", "full"))
        series, used, msg = None, "", ""
        with _wrap(self.name, symbol):
            for function, size in attempts:
                data = self._get({"function": function, "symbol": code, "outputsize": size,
                                  "apikey": key})
                series = data.get("Time Series (Daily)")
                if series:
                    used = function
                    break
                if "Error Message" in data or "Note" in data:  # bad symbol / rate limit
                    raise DataSourceError(f"Alpha Vantage: {data.get('Error Message') or data['Note']}")
                msg = str(data.get("Information", data))  # premium-only endpoint: degrade
            if not series:
                raise DataSourceError(f"Alpha Vantage 未返回 {code} 数据: {msg}")
            df = pd.DataFrame.from_dict(series, orient="index")
            df.columns = [c.split(". ", 1)[-1] for c in df.columns]
            df = df.apply(pd.to_numeric, errors="coerce")
            if "adjusted close" in df:  # scale OHLC by the adjustment factor
                f = df["adjusted close"] / df["close"]
                df[["open", "high", "low"]] = df[["open", "high", "low"]].mul(f, axis=0)
                df["close"] = df["adjusted close"]
            else:
                log.warning("Alpha Vantage 使用 %s（未复权）获取 %s", used, symbol)
            bars = to_bars(df)
        return clip_dates(bars, start, end)


# ----------------------------------------------------------------- CSV
class CSVSource(DataSource):
    """Local CSV files: ``<dir>/<SYMBOL>.csv`` (daily) or ``<dir>/<SYMBOL>_<freq>.csv``.

    Accepts a ``date``/``datetime`` column (or Chinese 日期/时间) plus OHLCV, with
    English or Chinese headers (开盘/最高/最低/收盘/成交量/成交额). ``volume_multiplier``
    converts e.g. 手 to shares (100).
    """

    name = "csv"
    markets = ("CN", "HK", "US")
    frequencies = ALL_FREQS
    cacheable = False

    def __init__(self, directory: str | Path = "data", volume_multiplier: float = 1.0):
        self.directory = Path(directory)
        self.volume_multiplier = volume_multiplier

    def path_for(self, symbol: str, frequency: str = "1d") -> Optional[Path]:
        sym = normalize_symbol(symbol)
        code = sym.rsplit(".", 1)[0]
        stems = [f"{sym}_{frequency}"] + ([sym, code] if frequency == "1d" else [])
        for stem in stems:
            for name in (stem, stem.lower()):
                p = self.directory / f"{name}.csv"
                if p.exists():
                    return p
        return None

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        p = self.path_for(symbol, frequency)
        if p is None:
            raise DataSourceError(f"找不到 CSV 文件: {self.directory / normalize_symbol(symbol)}.csv")
        try:
            df = pd.read_csv(p, encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            raise DataSourceError(f"读取 {p} 失败: {e}") from e
        bars = parse_bar_frame(df)
        bars["volume"] = bars["volume"] * self.volume_multiplier
        return clip_dates(bars, start, end)

    def save(self, symbol: str, df: pd.DataFrame, frequency: str = "1d") -> Path:
        """Write bars so that ``fetch`` reads them back unchanged."""
        self.directory.mkdir(parents=True, exist_ok=True)
        sym = normalize_symbol(symbol)
        p = self.directory / (f"{sym}.csv" if frequency == "1d" else f"{sym}_{frequency}.csv")
        out = df.copy()
        out.index.name = "datetime"
        out.to_csv(p)
        return p


# ----------------------------------------------------------------- Synthetic
class SyntheticSource(DataSource):
    """Deterministic, realistic-looking daily bars for any symbol (offline demos/tests).

    Each symbol gets its own random streams seeded by ``(seed, crc32(symbol))``;
    the series is generated on a fixed business-day calendar from 2005-01-03 so
    any date's bar is independent of the requested range. Prices follow a
    regime-switching GBM (bull / bear / mean-reverting range) with volatility
    clustering; A-share symbols respect their daily price limits (±10%/20%/30%).
    """

    name = "synthetic"
    markets = ("CN", "HK", "US")
    frequencies = ("1d",)
    cacheable = False
    ANCHOR = pd.Timestamp("2005-01-03")
    # (daily drift, daily vol, mean-reversion speed)
    REGIMES = ((0.0009, 0.014, 0.0), (-0.0008, 0.020, 0.0), (0.0, 0.011, 0.04))
    PRICE_RANGE = {"CN": (3.0, 150.0), "HK": (1.0, 400.0), "US": (10.0, 400.0)}

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._memo: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        if frequency != "1d":
            raise DataSourceError(f"合成数据源仅支持日线，不支持 {frequency}")
        today = pd.Timestamp.today().normalize()
        end_ts = min(pd.Timestamp(end).normalize(), today) if end else today
        sym = normalize_symbol(symbol)
        key = (sym, end_ts)
        if key not in self._memo:
            if len(self._memo) > 256:
                self._memo.clear()
            self._memo[key] = self.generate(sym, end_ts)
        bars = clip_dates(self._memo[key], start, end)
        if bars.empty:
            raise DataSourceError(f"合成数据在 {start}~{end} 区间为空（起点 {self.ANCHOR.date()}）")
        return bars.copy()

    def generate(self, symbol: str, end: pd.Timestamp) -> pd.DataFrame:
        sym = normalize_symbol(symbol)
        mkt = market_of(sym)
        days = pd.bdate_range(self.ANCHOR, end)
        n = len(days)
        h = zlib.crc32(sym.encode("utf-8"))

        def rng(stream: int) -> np.random.Generator:  # independent prefix-stable streams
            return np.random.default_rng([self.seed & 0xFFFFFFFF, h, stream])

        meta = rng(0)
        lo, hi = self.PRICE_RANGE.get(mkt, (10.0, 400.0))
        p0 = float(np.exp(meta.uniform(np.log(lo), np.log(hi))))
        vol_scale = meta.uniform(0.7, 1.4)
        base_volume = 10 ** meta.uniform(5.5, 7.5)
        eps, switch, gap = rng(1).standard_normal(n), rng(2).random(n), rng(3).standard_normal(n)
        wick, vnoise = np.abs(rng(4).standard_normal((n, 2))), rng(5).standard_normal(n)
        jump = rng(6).integers(1, 3, n)

        rules = a_share_spec().rules if mkt == "CN" else None
        decimals = 3 if mkt == "HK" else 2
        tick = 10.0 ** -decimals
        o = np.empty(n); hgh = np.empty(n); low = np.empty(n); c = np.empty(n); vol = np.empty(n)
        prev, regime, v = round(p0, decimals), 0, 1.0
        long_level = anchor = math.log(p0)
        for i in range(n):
            if switch[i] > 0.985:  # ~66-day average regime length
                regime = (regime + int(jump[i])) % 3
                anchor = math.log(prev)
            mu, sig, kappa = self.REGIMES[regime]
            sig *= vol_scale * v
            lp = math.log(prev)
            ret = mu + kappa * (anchor - lp) + 0.001 * (long_level - lp) + sig * eps[i]
            v = 0.94 * v + 0.06 * abs(eps[i]) * 1.2533  # volatility clustering, mean ~1
            close = prev * math.exp(ret)
            opn = prev * math.exp(0.3 * sig * gap[i])
            high = max(opn, close) * math.exp(0.5 * sig * wick[i, 0])
            lw = min(opn, close) * math.exp(-0.5 * sig * wick[i, 1])
            vals = [max(round(x, decimals), tick) for x in (opn, high, lw, close)]
            if rules is not None:
                lim_lo, lim_hi = rules.limit_prices(sym, prev)
                vals = [min(max(x, lim_lo), lim_hi) for x in vals]
            opn, high, lw, close = vals
            o[i], c[i] = opn, close
            hgh[i], low[i] = max(high, opn, close), min(lw, opn, close)
            vol[i] = base_volume * math.exp(0.35 * vnoise[i]) * (0.5 + 0.5 * v + 0.4 * abs(eps[i]))
            prev = close
        lot = 100 if mkt in ("CN", "HK") else 1
        vol = np.maximum(np.round(vol / lot), 1) * lot
        df = pd.DataFrame({"open": o, "high": hgh, "low": low, "close": c, "volume": vol},
                          index=pd.DatetimeIndex(days, name="datetime"))
        df["amount"] = df["volume"] * (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        return df
