# -*- coding: utf-8 -*-
"""
报告生成模块
负责生成各种分析报告和可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def generate_weights_display(weights_code, code2name):
    """生成权重展示表"""
    weights_display = pd.Series(
        {code2name[c]: w for c, w in weights_code.items()},
        name='Weight'
    ).to_frame()
    return weights_display


def generate_summary_report(backtest_results, alpha_annual, beta, ytd_ret, t12m_ret):
    """生成汇总指标报告"""
    summary = {
        'Total Annualized Return': backtest_results['ann_return'],
        'Total Return': backtest_results['total_return'],
        'Sharpe (Rf=3%)': backtest_results['sharpe'],
        'Calm Ratio': backtest_results['calm_ratio'],
        'Volatility (ann.)': backtest_results['vol_ann'],
        'Max Drawdown': backtest_results['max_drawdown'],
        'Alpha (annual)': alpha_annual,
        'Beta': beta,
        'Trades': backtest_results['num_trades'],
        'Win Rate': backtest_results['win_rate'],
        'YTD': ytd_ret,
        'T12M': t12m_ret
    }
    return pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])


def generate_quarterly_funds_report(close_df_code, code2name):
    """生成季度基金选择报告"""
    quarters_list = sorted(set(close_df_code.index.to_period('Q').astype(str)))
    selected_each_quarter = pd.Series(
        {q: [code2name[c] for c in close_df_code.columns] for q in quarters_list}
    )
    return pd.DataFrame({'Quarter': quarters_list, 'Selected': selected_each_quarter.values})


def display_all_reports(backtest_results, weights_display, summary_df, yearly_df, 
                       quarterly_returns, chosen_funds_df, trade_records, period_returns):
    """显示所有报告"""
    display(weights_display)
    display(summary_df)
    
    print("\n== Yearly Stats ==")
    display(yearly_df)
    
    print("\n== Quarterly Returns (quarter-end only) ==")
    display(quarterly_returns)
    
    print("\n== Quarterly Selected Funds ==")
    display(chosen_funds_df)
    
    print("\n== Trade Records ==")
    display(trade_records)
    
    print("\n== Period Returns (between rebalances) ==")
    display(period_returns)


def plot_equity_curve(equity_curve):
    """绘制净值曲线图"""
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve.index, equity_curve.values, label='Equity')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Net Asset Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_drawdown(equity_curve):
    """绘制回撤图"""
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    plt.figure(figsize=(10, 3.5))
    plt.plot(drawdown.index, drawdown.values, label='Drawdown')
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    plt.show()


def generate_complete_report(backtest_results, weights_code, code2name, close_df_code, 
                           alpha_annual, beta, ytd_ret, t12m_ret):
    """生成完整报告"""
    # 生成各种报告表格
    weights_display = generate_weights_display(weights_code, code2name)
    summary_df = generate_summary_report(backtest_results, alpha_annual, beta, ytd_ret, t12m_ret)
    chosen_funds_df = generate_quarterly_funds_report(close_df_code, code2name)
    
    # 显示所有报告
    display_all_reports(
        backtest_results, weights_display, summary_df, 
        backtest_results.get('yearly_stats', pd.DataFrame()),
        backtest_results['quarterly_returns'], chosen_funds_df,
        backtest_results['trade_records'], backtest_results['period_returns']
    )
    
    # 绘制图表
    plot_equity_curve(backtest_results['equity_curve'])
    plot_drawdown(backtest_results['equity_curve'])
    
    return {
        'weights_display': weights_display,
        'summary_df': summary_df,
        'chosen_funds_df': chosen_funds_df
    }


def export_results_to_csv(backtest_results, weights_code, code2name, filename_prefix):
    """导出结果到CSV文件"""
    # 导出净值曲线
    backtest_results['equity_curve'].to_csv(f"{filename_prefix}_equity_curve.csv")
    
    # 导出权重
    weights_df = pd.Series(weights_code).to_frame('Weight')
    weights_df.index.name = 'Code'
    weights_df.to_csv(f"{filename_prefix}_weights.csv")
    
    # 导出交易记录
    backtest_results['trade_records'].to_csv(f"{filename_prefix}_trades.csv", index=False)
    
    # 导出季度收益
    backtest_results['quarterly_returns'].to_csv(f"{filename_prefix}_quarterly_returns.csv", index=False)
    
    print(f"结果已导出到 {filename_prefix}_*.csv 文件")


def generate_performance_summary(backtest_results, alpha_annual, beta):
    """生成性能摘要"""
    summary = f"""
性能摘要:
==========
年化收益率: {backtest_results['ann_return']:.2%}
总收益率: {backtest_results['total_return']:.2%}
夏普比率: {backtest_results['sharpe']:.2f}
宁静比率 (新): {backtest_results['calm_ratio']:.2f}
SPY宁静比率: {backtest_results['spy_calm_ratio']:.2f}
年化波动率: {backtest_results['vol_ann']:.2%}
最大回撤: {backtest_results['max_drawdown']:.2%}
CDaR (5%): {backtest_results['cdar']:.2%}
溃疡指数: {backtest_results['ulcer_index']:.2%}
Alpha: {alpha_annual:.2%}
Beta: {beta:.2f}
交易次数: {backtest_results['num_trades']}
胜率: {backtest_results['win_rate']:.2%}
    """
    return summary
