# -*- coding: utf-8 -*-
"""
配置文件模块
包含全局配置参数和资产映射
"""

# ================= Global Config =================
START_DATE = '2021-07-01'
END_DATE = '2025-10-13'
INITIAL_CASH = 10000.0
COMMISSION = 0.001  # 0.1% per trade
RF_ANNUAL = 0.03  # risk-free for Sharpe / alpha
DATA_DIR = "./data/20210701_20251013"

# 资产映射：股票代码 -> 描述名称（展示用）
ASSET_NAME_MAP = {
    '516780': 'EARTH_ETF',
    '159687': 'YATAI_ETF',
    '513880': 'RIJIN_ETF',
    '513310': 'BANDAO_ETF',
    # '513730': 'DONGNAN_ETF',
    '159726': 'HSHL_ETF',
    '159934': 'GLD_ETF',
    '511260': 'GUOZAI_ETF',
    '159941': 'NASDAQ_ETF',
    'SPY': 'SPY_ETF'
}

# 自定义权重（按股票代码填写）
WEIGHTS_USER = {
    '516780': 4,
    '513880': 3,
    '513310': 2,
    # # '513730': 1,
    # '159687': 8,
    '159726': 11,
    '159934': 30,
    '511260': 20,
    '159941': 20,
    'SPY': 10
}
