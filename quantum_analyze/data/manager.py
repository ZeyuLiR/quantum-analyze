"""DataManager: per-market source priority, automatic fallback, local cache, cross-validation.

Example config (``DataManager.from_config``)::

    {"cache_dir": "data_cache", "validate": False, "tolerance": 0.02,
     "sources": {"CN": ["akshare", "baostock", "tushare", "qmt"],
                 "HK": ["futu", "yfinance", "ibkr"],
                 "US": ["yfinance", "ibkr", "alphavantage", "futu"]},
     "options": {"tushare": {"token": "..."}, "futu": {"host": "127.0.0.1", "port": 11111},
                 "ibkr": {"host": "127.0.0.1", "port": 7497, "client_id": 17},
                 "csv": {"path": "data"}, "synthetic": {"seed": 42}}}

``"sources": "synthetic"`` (a single name or a list) applies to every market.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import pandas as pd

from ..core.rules import market_of, normalize_symbol
from .base import DataSource, DataSourceError, DataSourceUnavailable, clip_dates, normalize_bars
from .sources import (ALL_FREQS, AkShareSource, AlphaVantageSource, BaoStockSource, CSVSource,
                      FutuSource, IBKRSource, QMTSource, SyntheticSource, TushareSource,
                      YFinanceSource)

log = logging.getLogger(__name__)

MARKETS = ("CN", "HK", "US")

SOURCE_REGISTRY: dict[str, type[DataSource]] = {
    "akshare": AkShareSource, "baostock": BaoStockSource, "tushare": TushareSource,
    "qmt": QMTSource, "futu": FutuSource, "yfinance": YFinanceSource, "ibkr": IBKRSource,
    "alphavantage": AlphaVantageSource, "csv": CSVSource, "synthetic": SyntheticSource,
}
ALIASES = {"miniqmt": "qmt", "xtquant": "qmt", "yahoo": "yfinance", "yf": "yfinance",
           "ib": "ibkr", "av": "alphavantage", "alpha_vantage": "alphavantage"}
# config option spellings -> constructor argument names
_ARG_ALIASES = {"csv": {"path": "directory", "dir": "directory"},
                "ibkr": {"clientId": "client_id"},
                "alphavantage": {"key": "api_key", "apikey": "api_key"}}

SourceSpec = Union[DataSource, str, Sequence[Union[DataSource, str]]]


def build_source(name: str, **kwargs: Any) -> DataSource:
    """Instantiate a source by name (akshare, baostock, tushare, qmt, futu, yfinance,
    ibkr, alphavantage, csv, synthetic). Constructors never touch the network."""
    key = ALIASES.get(name.strip().lower(), name.strip().lower())
    if key not in SOURCE_REGISTRY:
        raise ValueError(f"未知数据源: {name}（可选: {', '.join(SOURCE_REGISTRY)}）")
    alias = _ARG_ALIASES.get(key, {})
    kwargs = {alias.get(k, k): v for k, v in kwargs.items()}
    try:
        return SOURCE_REGISTRY[key](**kwargs)
    except TypeError as e:
        raise ValueError(f"数据源 {key} 的参数无效: {e}") from e


# ----------------------------------------------------------------- validation
@dataclass
class ValidationReport:
    symbol: str
    source_a: str
    source_b: str
    overlap: int
    mismatches: int
    mismatch_ratio: float
    max_deviation: float
    mean_deviation: float
    worst_date: Optional[str]
    tolerance: float

    @property
    def ok(self) -> bool:
        return self.overlap > 0 and self.mismatches == 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}

    def __str__(self) -> str:
        return (f"{self.symbol} {self.source_a} vs {self.source_b}: 重叠 {self.overlap} 根, "
                f"偏差>{self.tolerance:.1%} 的 {self.mismatches} 根 ({self.mismatch_ratio:.1%}), "
                f"最大偏差 {self.max_deviation:.2%} @ {self.worst_date}")


def compare_bars(a: pd.DataFrame, b: pd.DataFrame, tolerance: float = 0.02, symbol: str = "",
                 names: tuple[str, str] = ("a", "b"), daily: bool = True) -> ValidationReport:
    """Compare close prices on overlapping dates; deviation is relative to ``b``."""
    ca, cb = a["close"], b["close"]
    if daily:
        ca = ca.groupby(ca.index.normalize()).last()
        cb = cb.groupby(cb.index.normalize()).last()
    common = ca.index.intersection(cb.index)
    dev = ((ca.loc[common] - cb.loc[common]).abs() / cb.loc[common].abs()).dropna()
    n = len(dev)
    mism = int((dev > tolerance).sum())
    return ValidationReport(
        symbol=symbol, source_a=names[0], source_b=names[1], overlap=n, mismatches=mism,
        mismatch_ratio=mism / n if n else 0.0,
        max_deviation=float(dev.max()) if n else 0.0,
        mean_deviation=float(dev.mean()) if n else 0.0,
        worst_date=str(dev.idxmax().date()) if n else None, tolerance=tolerance)


def cross_validate(symbol: str, sources: Sequence[DataSource], start: Optional[str] = None,
                   end: Optional[str] = None, tolerance: float = 0.02,
                   frequency: str = "1d") -> ValidationReport:
    """Fetch ``symbol`` from two sources and compare closes (no caching)."""
    if len(sources) != 2:
        raise ValueError("cross_validate 需要恰好两个数据源")
    sym = normalize_symbol(symbol)
    a, b = (normalize_bars(s.fetch(sym, start, end, frequency)) for s in sources)
    return compare_bars(a, b, tolerance, sym, (sources[0].name, sources[1].name),
                        daily=frequency == "1d")


# ----------------------------------------------------------------- cache
class BarCache:
    """Per (source, frequency, symbol) bar files plus a JSON sidecar with the covered range."""

    def __init__(self, root: str | Path, fmt: str = "auto"):
        self.root = Path(root)
        if fmt == "auto":
            fmt = "parquet" if self._has_parquet() else "csv"
        if fmt not in ("csv", "parquet"):
            raise ValueError(f"不支持的缓存格式: {fmt}")
        self.fmt = fmt

    @staticmethod
    def _has_parquet() -> bool:
        for mod in ("pyarrow", "fastparquet"):
            try:
                __import__(mod)
                return True
            except ImportError:
                continue
        return False

    def _paths(self, source: str, symbol: str, frequency: str) -> tuple[Path, Path]:
        d = self.root / source / frequency
        return d / f"{symbol}.{self.fmt}", d / f"{symbol}.meta.json"

    def read(self, source: str, symbol: str, frequency: str) -> tuple[Optional[pd.DataFrame], dict]:
        data, meta = self._paths(source, symbol, frequency)
        if not data.exists() or not meta.exists():
            return None, {}
        try:
            if self.fmt == "parquet":
                df = pd.read_parquet(data)
            else:
                df = pd.read_csv(data, index_col=0, parse_dates=[0])
            return normalize_bars(df), json.loads(meta.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - a corrupt cache is just a miss
            log.warning("缓存文件损坏，忽略: %s (%s)", data, e)
            return None, {}

    def write(self, source: str, symbol: str, frequency: str, df: pd.DataFrame,
              covered_start: Optional[pd.Timestamp], covered_end: pd.Timestamp) -> None:
        data, meta = self._paths(source, symbol, frequency)
        data.parent.mkdir(parents=True, exist_ok=True)
        if self.fmt == "parquet":
            df.to_parquet(data)
        else:
            df.to_csv(data)
        meta.write_text(json.dumps({
            "start": covered_start.strftime("%Y-%m-%d") if covered_start is not None else None,
            "end": covered_end.strftime("%Y-%m-%d"), "rows": len(df)}), encoding="utf-8")


# ----------------------------------------------------------------- manager
class DataManager(DataSource):
    """Loads bars through an ordered list of sources per market with fallback.

    * ``provenance[symbol]`` records which source served each symbol;
      ``errors[symbol]`` the failures of the sources tried before it.
    * With ``cache_dir`` set, fetched bars are cached per source/frequency/symbol and
      later loads only fetch the missing tail (re-downloading everything if the
      adjustment factors changed). ``refresh=True`` bypasses the cache.
    * With ``validate=True`` each symbol is also fetched from the next available
      source and compared (``validation[symbol]``); mismatches are logged as warnings.
    """

    name = "manager"
    markets = MARKETS
    frequencies = ALL_FREQS

    def __init__(self, sources: Union[SourceSpec, dict[str, SourceSpec], None] = None,
                 cache_dir: Optional[str | Path] = None, validate: bool = False,
                 tolerance: float = 0.02, cache_format: str = "auto", tail_overlap: int = 5,
                 options: Optional[dict[str, dict[str, Any]]] = None):
        self.options = options or {}
        self._built: dict[str, DataSource] = {}
        spec = sources if sources is not None else "synthetic"
        if not isinstance(spec, dict):
            spec = {m: spec for m in MARKETS}
        self.sources: dict[str, list[DataSource]] = {
            str(m).upper(): self._resolve(v) for m, v in spec.items()}
        self.cache = BarCache(cache_dir, cache_format) if cache_dir else None
        self.validate = validate
        self.tolerance = tolerance
        self.tail_overlap = max(int(tail_overlap), 1)
        self.provenance: dict[str, str] = {}
        self.errors: dict[str, dict[str, str]] = {}
        self.validation: dict[str, ValidationReport] = {}

    # -- construction
    def _get(self, name: str) -> DataSource:
        key = ALIASES.get(name.strip().lower(), name.strip().lower())
        if key not in self._built:
            self._built[key] = build_source(key, **self.options.get(key, {}))
        return self._built[key]

    def _resolve(self, spec: SourceSpec) -> list[DataSource]:
        items = [spec] if isinstance(spec, (str, DataSource)) else list(spec)
        return [self._get(s) if isinstance(s, str) else s for s in items]

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]] = None) -> "DataManager":
        cfg = config or {}
        return cls(cfg.get("sources", "synthetic"), cache_dir=cfg.get("cache_dir"),
                   validate=bool(cfg.get("validate", False)),
                   tolerance=float(cfg.get("tolerance", 0.02)),
                   cache_format=cfg.get("cache_format", "auto"),
                   options={ALIASES.get(k.lower(), k.lower()): dict(v or {})
                            for k, v in (cfg.get("options") or {}).items()})

    # -- querying
    def sources_for(self, symbol: str, frequency: str = "1d") -> list[DataSource]:
        mkt = market_of(normalize_symbol(symbol))
        return [s for s in self.sources.get(mkt, []) if s.supports(symbol, frequency)]

    def supports(self, symbol: str, frequency: str = "1d") -> bool:
        return bool(self.sources_for(symbol, frequency))

    def fetch(self, symbol: str, start: Optional[str], end: Optional[str],
              frequency: str = "1d") -> pd.DataFrame:
        return self._load_one(normalize_symbol(symbol), start, end, frequency, False)

    def load(self, symbols: Union[str, Iterable[str]], start: Optional[str] = None,
             end: Optional[str] = None, frequency: str = "1d", refresh: bool = False,
             errors: str = "raise") -> dict[str, pd.DataFrame]:
        """Load bars for each symbol. ``errors="skip"`` logs and omits failed symbols."""
        symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        out: dict[str, pd.DataFrame] = {}
        for s in symbols:
            sym = normalize_symbol(s)
            try:
                out[sym] = self._load_one(sym, start, end, frequency, refresh)
            except DataSourceError as e:
                if errors != "skip":
                    raise
                log.error("跳过 %s: %s", sym, e)
        return out

    def _load_one(self, sym: str, start: Optional[str], end: Optional[str],
                  frequency: str, refresh: bool) -> pd.DataFrame:
        candidates = self.sources_for(sym, frequency)
        if not candidates:
            raise DataSourceUnavailable(
                f"{sym} 没有可用的数据源（市场 {market_of(sym)}，频率 {frequency}），请检查 data.sources 配置")
        failures: dict[str, str] = {}
        for i, src in enumerate(candidates):
            try:
                df = self._fetch_cached(src, sym, start, end, frequency, refresh)
            except DataSourceError as e:
                failures[src.name] = str(e)
                level = logging.INFO if isinstance(e, DataSourceUnavailable) else logging.WARNING
                log.log(level, "%s 获取 %s 失败，尝试下一个数据源: %s", src.name, sym, e)
                continue
            self.provenance[sym] = src.name
            self.errors[sym] = failures
            log.info("%s 数据来自 %s（%d 根 %s K线）", sym, src.name, len(df), frequency)
            if self.validate:
                self._auto_validate(sym, df, src, candidates[i + 1:], start, end, frequency, refresh)
            return df
        self.errors[sym] = failures
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items())
        raise DataSourceError(f"{sym} 所有数据源均失败 — {detail}")

    # -- cache
    def _fetch_cached(self, src: DataSource, sym: str, start: Optional[str], end: Optional[str],
                      frequency: str, refresh: bool) -> pd.DataFrame:
        if self.cache is None or not getattr(src, "cacheable", True):
            return normalize_bars(src.fetch(sym, start, end, frequency))
        today = pd.Timestamp.today().normalize()
        req_start = pd.Timestamp(start).normalize() if start else None
        req_end = min(pd.Timestamp(end).normalize(), today) if end else today
        # today's bars may still change: never mark today as covered
        covered_end = req_end if req_end < today else today - pd.Timedelta(days=1)

        cached, meta = (None, {}) if refresh else self.cache.read(src.name, sym, frequency)
        if cached is not None and not cached.empty:
            c_start = pd.Timestamp(meta["start"]) if meta.get("start") else None
            c_end = pd.Timestamp(meta["end"])
            head_ok = c_start is None or (req_start is not None and req_start >= c_start)
            if head_ok and req_end <= c_end:
                log.debug("%s %s 命中缓存 (%s)", sym, frequency, src.name)
                return self._clip(cached, start, end, sym)
            if head_ok:
                merged = self._extend_tail(src, sym, cached, end, frequency)
                if merged is not None:
                    self.cache.write(src.name, sym, frequency, merged, c_start,
                                     max(c_end, covered_end))
                    return self._clip(merged, start, end, sym)
            # head missing or adjustment changed: refetch the union of both ranges
            if req_start is not None and c_start is not None:
                req_start = min(req_start, c_start)
            elif c_start is None:
                req_start = None
            if end and c_end > req_end:
                end_arg: Optional[str] = c_end.strftime("%Y-%m-%d")
                covered_end = c_end
            else:
                end_arg = end
        else:
            end_arg = end
        df = normalize_bars(src.fetch(
            sym, req_start.strftime("%Y-%m-%d") if req_start is not None else None, end_arg,
            frequency))
        self.cache.write(src.name, sym, frequency, df, req_start, covered_end)
        return self._clip(df, start, end, sym)

    def _extend_tail(self, src: DataSource, sym: str, cached: pd.DataFrame, end: Optional[str],
                     frequency: str) -> Optional[pd.DataFrame]:
        """Fetch from a few bars before the cached end; None if history was re-adjusted."""
        tail_from = cached.index[-min(self.tail_overlap, len(cached))].normalize()
        tail = normalize_bars(src.fetch(sym, tail_from.strftime("%Y-%m-%d"), end, frequency))
        common = cached.index.intersection(tail.index)
        if len(common) == 0:
            log.info("%s 增量数据与缓存无重叠，重新下载全量", sym)
            return None
        a, b = cached.loc[common, "close"], tail.loc[common, "close"]
        if ((a - b).abs() > 1e-6 * b.abs().clip(lower=1e-9)).any():
            log.info("%s 复权价格已变化（可能发生除权），重新下载全量 (%s)", sym, src.name)
            return None
        merged = pd.concat([cached[cached.index < tail.index[0]], tail])
        merged["pre_close"] = merged["pre_close"].fillna(merged["close"].shift(1))
        return merged

    @staticmethod
    def _clip(df: pd.DataFrame, start: Optional[str], end: Optional[str], sym: str) -> pd.DataFrame:
        out = clip_dates(df, start, end)
        if out.empty:
            raise DataSourceError(f"{sym} 在 {start}~{end} 区间没有数据")
        return out

    # -- validation
    def _auto_validate(self, sym: str, df: pd.DataFrame, src: DataSource,
                       rest: list[DataSource], start: Optional[str], end: Optional[str],
                       frequency: str, refresh: bool) -> None:
        for alt in rest:
            try:
                other = self._fetch_cached(alt, sym, start, end, frequency, refresh)
            except DataSourceError as e:
                log.debug("交叉验证源 %s 不可用: %s", alt.name, e)
                continue
            rep = compare_bars(df, other, self.tolerance, sym, (src.name, alt.name),
                               daily=frequency == "1d")
            self.validation[sym] = rep
            if rep.ok:
                log.info("交叉验证通过: %s", rep)
            else:
                log.warning("交叉验证不一致: %s", rep)
            return
        log.info("%s 没有可用于交叉验证的备用数据源", sym)

    def cross_validate(self, symbol: str, sources: Optional[Sequence[Union[DataSource, str]]] = None,
                       start: Optional[str] = None, end: Optional[str] = None,
                       tolerance: Optional[float] = None, frequency: str = "1d") -> ValidationReport:
        """Compare closes from two sources (names or instances; default: first two configured)."""
        sym = normalize_symbol(symbol)
        srcs = self._resolve(sources) if sources is not None else self.sources_for(sym, frequency)[:2]
        if len(srcs) != 2:
            raise ValueError(f"{sym} 交叉验证需要两个数据源，当前 {len(srcs)} 个")
        a, b = (self._fetch_cached(s, sym, start, end, frequency, False) for s in srcs)
        rep = compare_bars(a, b, self.tolerance if tolerance is None else tolerance, sym,
                           (srcs[0].name, srcs[1].name), daily=frequency == "1d")
        self.validation[sym] = rep
        return rep
