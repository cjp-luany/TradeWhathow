# -*- coding: utf-8 -*-
"""
宁静比率计算模块
使用新公式：宁静比率 = (年化收益率 - 无风险收益率) / √(CDaR × Ulcer Index)
"""

import numpy as np
import pandas as pd


def calculate_serenity_ratio_new(equity_curve, daily_ret, rf_rate=0.025, alpha=0.05):
    """
    使用修正后的溃疡指数计算宁静比率
    """
    # 1. 计算年化收益率
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    trading_days = len(daily_ret)
    ann_return = (1.0 + total_return) ** (252.0 / trading_days) - 1.0
    
    # 2. 计算CDaR
    cdar = calculate_cdar(equity_curve, alpha)
    
    # 3. 计算溃疡指数（使用修正后的方法）
    ulcer_index = calculate_ulcer_index_corrected(equity_curve)
    
    # 4. 计算宁静比率
    excess_return = ann_return - rf_rate
    denominator = np.sqrt(cdar * ulcer_index)
    
    if denominator > 1e-10:
        serenity_ratio = excess_return / denominator
    else:
        serenity_ratio = np.nan
    
    # 输出调试信息
    print(f"年化收益率: {ann_return:.4%}")
    print(f"CDaR ({alpha:.0%}): {cdar:.4%}")
    print(f"溃疡指数: {ulcer_index:.4%}")
    print(f"分母: √({cdar:.4%} × {ulcer_index:.4%}) = {denominator:.6f}")
    print(f"宁静比率: {serenity_ratio:.4f}")
    
    return serenity_ratio, ann_return, cdar, ulcer_index


def calculate_cdar(equity_curve, alpha=0.05):
    """
    计算条件在险回撤 (Conditional Drawdown at Risk)
    在给定置信水平alpha下的平均最大回撤
    """
    # 计算回撤序列
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    
    # 计算回撤的VaR (在险价值)
    var_threshold = np.percentile(drawdown, alpha * 100)
    
    # 计算条件在险回撤 (超过VaR的回撤的平均值)
    tail_drawdowns = drawdown[drawdown <= var_threshold]
    
    if len(tail_drawdowns) > 0:
        cdar = abs(tail_drawdowns.mean())  # 取绝对值转为正数
    else:
        cdar = abs(var_threshold)  # 如果没有超过VaR的回撤，使用VaR值
    
    return cdar


def calculate_ulcer_index_corrected(equity_curve):
    """
    正确计算溃疡指数 (Ulcer Index)
    溃疡指数 = sqrt(平均(回撤百分比^2))
    
    注意：溃疡指数是基于整个时间段计算的，不是滚动窗口
    """
    # 计算回撤百分比
    roll_max = equity_curve.cummax()
    drawdown_pct = (roll_max - equity_curve) / roll_max
    
    # 溃疡指数公式: sqrt(平均(回撤百分比^2))
    squared_drawdown = drawdown_pct ** 2
    
    # 计算整个时间段内平方回撤的平均值，然后开方
    ulcer_index = np.sqrt(squared_drawdown.mean())
    
    return ulcer_index

def debug_ulcer_calculation(equity_curve):
    """
    调试溃疡指数计算，验证每一步
    """
    print("=== 溃疡指数计算调试 ===")
    
    # 计算回撤序列
    roll_max = equity_curve.cummax()
    drawdown_pct = (roll_max - equity_curve) / roll_max
    
    print(f"净值曲线长度: {len(equity_curve)}")
    print(f"最大回撤: {drawdown_pct.min():.4%}")
    print(f"平均回撤: {drawdown_pct.mean():.4%}")
    
    # 计算平方回撤
    squared_drawdown = drawdown_pct ** 2
    print(f"平方回撤均值: {squared_drawdown.mean():.8f}")
    
    # 计算溃疡指数
    ulcer_index = np.sqrt(squared_drawdown.mean())
    print(f"溃疡指数: {ulcer_index:.4%}")
    
    # 对比新旧方法
    old_method = calculate_ulcer_index_old(equity_curve)
    new_method = calculate_ulcer_index_corrected(equity_curve)
    
    print(f"旧方法结果: {old_method:.4%}")
    print(f"新方法结果: {new_method:.4%}")
    
    return new_method

# 保留旧方法用于对比
def calculate_ulcer_index_old(equity_curve, period=14):
    """您原来的计算方法"""
    roll_max = equity_curve.cummax()
    drawdown_pct = (roll_max - equity_curve) / roll_max
    squared_drawdown = drawdown_pct ** 2
    ulcer_index = np.sqrt(squared_drawdown.rolling(window=period, min_periods=1).mean())
    return ulcer_index.iloc[-1]