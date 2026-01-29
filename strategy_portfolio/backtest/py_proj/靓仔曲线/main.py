# -*- coding: utf-8 -*-
"""
主执行脚本
整合所有模块，执行完整的回测流程
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import START_DATE, END_DATE, INITIAL_CASH, COMMISSION, RF_ANNUAL
from config import ASSET_NAME_MAP, WEIGHTS_USER
from data_loader import get_all_price_series, prepare_close_matrix
from backtest_engine import (normalize_weights_by_code, get_rebalance_dates, 
                           run_backtest, calc_alpha, yearly_stats)
from report_generator import generate_complete_report


def main():
    """主函数：执行完整的回测流程"""
    print("=" * 60)
    print("固定组合测试系统 - 模块化版本")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n1. 数据准备中...")
    codes = list(ASSET_NAME_MAP.keys())
    prices_dict = get_all_price_series(codes, START_DATE, END_DATE)
    close_code = prepare_close_matrix(prices_dict, START_DATE, END_DATE)
    print(f"数据准备完成，共 {len(close_code)} 个交易日")
    
    # 2. 权重归一化
    print("\n2. 权重归一化...")
    weights_code = normalize_weights_by_code(WEIGHTS_USER, codes)
    print("权重归一化完成")
    
    # 3. 获取再平衡日期
    print("\n3. 计算再平衡日期...")
    trade_dates = get_rebalance_dates(close_code)
    print(f"再平衡日期计算完成，共 {len(trade_dates)} 次再平衡")
    
    # 4. 执行回测
    print("\n4. 执行回测...")
    bt_results = run_backtest(close_code, weights_code, trade_dates, INITIAL_CASH, COMMISSION)
    print("回测执行完成")
    
    # 5. 计算Alpha和Beta
    print("\n5. 计算Alpha和Beta...")
    alpha_annual, beta = calc_alpha(bt_results['daily_return'])
    print("Alpha/Beta计算完成")
    
    # 6. 计算年度统计
    print("\n6. 计算年度统计...")
    yearly_df, ytd_ret, t12m_ret = yearly_stats(bt_results['equity_curve'])
    bt_results['yearly_stats'] = yearly_df
    print("年度统计计算完成")
    
    # 7. 生成报告
    print("\n7. 生成报告...")
    report_results = generate_complete_report(
        bt_results, weights_code, ASSET_NAME_MAP, close_code,
        alpha_annual, beta, ytd_ret, t12m_ret
    )
    print("报告生成完成")
    
    # 8. 性能摘要
    print("\n8. 性能摘要:")
    from report_generator import generate_performance_summary
    summary = generate_performance_summary(bt_results, alpha_annual, beta)
    print(summary)
    
    print("\n" + "=" * 60)
    print("回测流程完成！")
    print("=" * 60)
    
    return bt_results, report_results


if __name__ == "__main__":
    # 执行主函数
    bt_results, report_results = main()
    
    # 可选：导出结果到CSV
    # from report_generator import export_results_to_csv
    # export_results_to_csv(bt_results, weights_code, ASSET_NAME_MAP, "backtest_results")
