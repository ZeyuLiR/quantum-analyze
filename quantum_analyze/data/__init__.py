"""Data sources: AkShare, BaoStock, Tushare, miniQMT, Futu, yfinance, IBKR, Alpha Vantage, CSV, synthetic."""
from .base import DataSource, DataSourceError, DataSourceUnavailable, normalize_bars
from .manager import DataManager, build_source
from .sources import CSVSource, SyntheticSource

__all__ = ["DataSource", "DataSourceError", "DataSourceUnavailable", "normalize_bars",
           "DataManager", "build_source", "CSVSource", "SyntheticSource"]
