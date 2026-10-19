# quantum-analyze

中低频量化交易系统，面向个人投资者，在自己电脑上运行：**A股 + 港股 + 美股**，**回测 → 模拟盘 → 实盘**用同一份策略代码、同一份配置文件。

综合了 qlib、vn.py、backtrader、zipline、freqtrade、rqalpha、QUANTAXIS 等高星开源项目的长处（详见 [策略说明书 · 第 1 章](docs/策略说明书.md#1-系统总览集百家之长)）。

## 功能

| 模块 | 内容 |
|---|---|
| 数据 | AkShare、BaoStock、Tushare、miniQMT、富途 OpenD、yfinance、IBKR、Alpha Vantage、CSV、合成数据；按市场设优先级、失败自动切换、本地增量缓存、多源交叉校验 |
| 交易规则 | A股 T+1、100 股一手、涨跌停（主板/创业板/科创板/北交所/ST）、停牌、印花税/过户费/最低佣金；港股每手股数、双边印花税；美股按股收费；多币种汇率折算 |
| 策略 | 动量轮动、量价多因子、风险平价、双均线、MACD、海龟突破、RSI 反转、布林带回归、网格、买入持有；可自定义 |
| 风控 | 单票仓位上限、单笔金额上限、黑名单、止损/移动止损/止盈/时间止损、账户回撤熔断 |
| 回测 | 事件驱动，默认“今日收盘出信号、次日开盘成交”，无未来函数；HTML 报告（净值、回撤、月度收益、成交） |
| 寻优 | 网格搜索、滚动 walk-forward 样本外检验 |
| 模拟盘 | 内置撮合，无需账户；每天运行一次，状态持久化；与回测逐笔一致（有测试保证） |
| 实盘 | IBKR、富途、老虎、长桥、miniQMT、easytrader、Ptrade（导出委托）、手工 CSV；默认 dry-run |
| 看板 | `qa dashboard` 本地网页查看账户、持仓、委托和回测报告 |

## 快速开始

```bash
pip install -e .            # 只依赖 numpy + pandas
qa strategies               # 查看内置策略
qa backtest -c examples/quickstart.toml     # 离线合成数据，体验完整流程
```

用真实数据：

```bash
pip install -e ".[cn]"      # A股：akshare + baostock
pip install -e ".[global]"  # 港美股：yfinance + IBKR + 富途
qa backtest    -c examples/cn_etf_rotation.toml
qa walkforward -c examples/cn_etf_rotation.toml
qa paper       -c examples/cn_etf_rotation.toml   # 每个交易日收盘后运行
qa live        -c examples/us_risk_parity_ibkr.toml   # 默认只打印委托
qa dashboard   -c examples/cn_etf_rotation.toml
```

| 示例配置 | 市场 | 策略 | 执行 |
|---|---|---|---|
| `examples/quickstart.toml` | 合成数据 | 多因子 | 内置模拟 |
| `examples/cn_etf_rotation.toml` | A股 ETF | 动量轮动 | miniQMT |
| `examples/cn_multi_factor.toml` | A股个股 | 多因子 + 大盘择时 | miniQMT |
| `examples/hk_trend_futu.toml` | 港股 | 双均线 | 富途（模拟账户） |
| `examples/us_risk_parity_ibkr.toml` | 美股 ETF | 风险平价 | IBKR（模拟端口） |

## 文档

- [策略说明书](docs/策略说明书.md)：每个策略的原理、信号、参数、适用场景与风险，风控说明，研究与上线流程，自定义策略，实盘接入。
- [实盘接入说明](docs/实盘接入说明.md)：各券商/数据源的安装与配置要点，以及需要在你自己电脑上核对的接口细节。

## 项目结构

```
quantum_analyze/
  core/        objects.py 订单/成交/持仓 · rules.py 市场规则与费用 · broker.py 模拟撮合 · context.py 策略 API
  data/        base.py 接口 · sources.py 各数据源 · manager.py 降级/缓存/校验
  gateway/     base.py 网关接口与 LiveAccount · brokers.py 各券商
  strategies/  base.py 策略基类 · builtin.py 内置策略
  analysis/    metrics.py 绩效指标 · report.py HTML 报告
  engine.py    事件引擎   risk.py 风控   optimize.py 寻优   trader.py 模拟盘/实盘
  config.py    TOML 配置  cli.py 命令行  dashboard.py 本地看板
tests/         离线测试（python -m pytest）
```

## 免责声明

本项目仅供研究学习，不构成投资建议。量化交易有亏损风险，实盘前请充分测试，并自行核对券商规则与费用。
