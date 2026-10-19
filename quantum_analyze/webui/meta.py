"""Chinese descriptions shown in the UI (kept in sync with docs/策略说明书.md)."""

STRATEGY_INFO = {
    "momentum_rotation": {
        "order": 1,
        "summary": "定期在一篮子品种中买入近期走势最强的 1~3 个；全部走弱时转入避险资产或空仓。",
        "fit": "ETF、指数、行业轮动；有主线的趋势行情。新手首选。",
        "risk": "来回震荡、没有主线时容易追高杀跌。",
    },
    "multi_factor": {
        "order": 2,
        "summary": "每月按动量、低波动、短期反转等因子给股票打分，买入得分最高的 N 只等权持有。",
        "fit": "个股，股票池建议 30 只以上。",
        "risk": "因子会阶段性失效；个股有暴雷、停牌风险。",
    },
    "risk_parity": {
        "order": 3,
        "summary": "按波动率倒数分配股票、债券、黄金等资产，让各资产风险贡献接近，回撤低。",
        "fit": "股/债/金/商品 ETF 的长期配置，适合作为底仓。",
        "risk": "股债同时下跌时失效；收益通常低于纯股票。",
    },
    "dual_ma": {
        "order": 4,
        "summary": "短均线上穿长均线（金叉）买入，下穿（死叉）卖出。最经典的趋势跟踪。",
        "fit": "指数 ETF、大盘股的单边趋势行情。",
        "risk": "震荡市假信号多，胜率通常只有 30~40%。",
    },
    "macd_trend": {
        "order": 5,
        "summary": "MACD 柱由负转正且价格在长期均线上方时买入，转负或跌破均线时卖出。",
        "fit": "指数 ETF、大盘股的中级趋势。",
        "risk": "参数对品种敏感，震荡市信号多。",
    },
    "turtle": {
        "order": 6,
        "summary": "突破过去 N 日最高价买入，跌破 M 日最低价或 ATR 止损卖出，按 ATR 控制每笔风险。",
        "fit": "波动较大的 ETF、商品、股票的大级别突破。",
        "risk": "假突破多、胜率低；A股跳空时止损可能失效。",
    },
    "rsi_reversion": {
        "order": 7,
        "summary": "长期上升趋势中，RSI 超卖时买入，反弹或持有到期后卖出。",
        "fit": "大盘蓝筹、宽基 ETF 的回调买点。",
        "risk": "单笔赚得少，对成本敏感；趋势反转时会越跌越买，务必设止损。",
    },
    "bollinger_reversion": {
        "order": 8,
        "summary": "价格跌破布林下轨后重新站上时买入，回到中轨卖出。",
        "fit": "宽基 ETF 的震荡行情。",
        "risk": "单边下跌中会连续失败。",
    },
    "grid": {
        "order": 9,
        "summary": "以均线为中枢，价格每下跌一格加仓、每上涨一格减仓，在震荡中低买高卖。",
        "fit": "单只宽基 ETF 的区间震荡行情。",
        "risk": "单边下跌会满仓被套，单边上涨会过早卖光。",
    },
    "buy_hold": {
        "order": 10,
        "summary": "第一天等权买入全部品种并一直持有，用来做对照。",
        "fit": "任何品种，作为基准对照。",
        "risk": "没有任何风险控制。",
    },
}

PARAM_LABELS = {
    "lookback": "动量窗口（交易日）", "top_k": "持有数量", "score": "动量算法（r2_slope / return）",
    "rebalance": "调仓频率（D 日 / W 周 / M 月 / Q 季）", "abs_threshold": "绝对动量门槛",
    "safe_asset": "避险品种代码（可空）", "skip": "跳过最近 N 日",
    "w_momentum": "动量因子权重", "w_lowvol": "低波动因子权重", "w_reversal": "反转因子权重",
    "w_liquidity": "流动性因子权重", "mom_n": "动量窗口", "mom_skip": "动量跳过天数",
    "vol_n": "波动率窗口", "rev_n": "反转窗口", "liq_n": "流动性窗口",
    "market_ma": "大盘择时均线（0=关闭）", "market_symbol": "大盘择时品种（可空）",
    "target_vol": "目标波动率（0=关闭）", "max_weight": "单品种最大仓位",
    "fast": "快线周期", "slow": "慢线周期", "trend_filter": "长期趋势过滤均线（0=关闭）",
    "max_positions": "最多同时持有（0=不限）", "signal": "信号线周期", "filter_ma": "趋势过滤均线",
    "entry_n": "入场通道（日）", "exit_n": "离场通道（日）", "atr_n": "ATR 周期",
    "risk_per_trade": "单笔风险占总资产", "stop_atr": "止损距离（ATR 倍数）",
    "rsi_n": "RSI 周期", "buy_below": "RSI 低于此值买入", "sell_above": "RSI 高于此值卖出",
    "trend_ma": "趋势均线（0=关闭）", "max_hold": "最长持有天数", "n": "布林带周期",
    "k": "布林带宽度（标准差倍数）", "symbol": "交易品种（空=股票池第一个）",
    "center_n": "中枢均线周期", "step": "网格间距", "levels": "单边网格数",
    "base_weight": "中枢处仓位",
    # risk
    "max_position_weight": "单票最大仓位", "max_order_value": "单笔金额上限（0=不限）",
    "stop_loss": "止损比例（0=关闭）", "trailing_stop": "移动止损比例（0=关闭）",
    "take_profit": "止盈比例（0=关闭）", "max_holding_days": "最长持有天数（0=不限）",
    "max_drawdown_halt": "账户回撤熔断（0=关闭）", "halt_cooldown_days": "熔断后暂停天数",
}
