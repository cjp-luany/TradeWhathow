# -*- coding: utf-8 -*-
"""
回测引擎模块
包含回测逻辑、交易执行和绩效计算
"""

import pandas as pd
import numpy as np
from config import RF_ANNUAL
from serenity_ratio import calculate_serenity_ratio_new


def normalize_weights_by_code(weights_user: dict, codes: list) -> dict:
    """归一化用户权重"""
    w = pd.Series(weights_user, dtype=float).reindex(codes).fillna(0.0)
    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / len(codes)
        print("[Info] Provided weights sum to 0. Using equal-weight.")
    else:
        w = (w / s).clip(lower=0)
        w = w / w.sum()
    zeros = w[w == 0]
    if len(zeros) > 0:
        print(f"[Warn] Zero weight for: {list(zeros.index)}")
    return w.to_dict()


def first_trading_day_of_years(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """获取每年的第一个交易日"""
    df = pd.DataFrame(index=idx)
    df['Y'] = df.index.year
    firsts = df.groupby('Y').apply(lambda x: x.index.min())
    return pd.DatetimeIndex(firsts.values)


def next_trading_day(idx: pd.DatetimeIndex, dt: pd.Timestamp):
    """获取下一个交易日"""
    pos = idx.searchsorted(dt, side='right')
    return None if pos >= len(idx) else idx[pos]


def get_rebalance_dates(close_df_code: pd.DataFrame):
    """获取再平衡日期（每年第一个交易日后的第一个交易日）"""
    year_first_days = first_trading_day_of_years(close_df_code.index)
    trade_dates = pd.DatetimeIndex(sorted(set(filter(
        None, (next_trading_day(close_df_code.index, yd) for yd in year_first_days)
    ))))
    trade_dates = trade_dates[
        (trade_dates >= close_df_code.index.min()) & 
        (trade_dates <= close_df_code.index.max())
    ]
    return trade_dates


def run_backtest(close_df_code: pd.DataFrame, weights_code: dict,
                 trade_dates: pd.DatetimeIndex,
                 initial_cash=10000.0, commission=0.001):
    """
    执行回测
    - 固定权重策略
    - 年度再平衡
    - 考虑交易费用
    """
    codes = list(close_df_code.columns)
    w_vec = pd.Series(weights_code).reindex(codes).fillna(0.0)

    cash = initial_cash
    shares = pd.Series(0.0, index=codes)
    equity_curve = pd.Series(index=close_df_code.index, dtype=float)
    positions_value = pd.Series(index=close_df_code.index, dtype=float)

    trade_records, period_pnls = [], []
    last_trade_nav, last_trade_date = None, None

    for dt in close_df_code.index:
        prices = close_df_code.loc[dt, codes]

        if dt in trade_dates:
            total_equity = cash + np.nansum(shares.values * prices.values)
            target_value = total_equity * w_vec
            current_value = shares * prices
            trade_value = target_value - current_value
            turnover = float(np.nansum(np.abs(trade_value.values)))
            cost = turnover * commission

            cash = cash - np.nansum(trade_value.values) - cost
            price_nonzero = prices.replace(0, np.nan)
            shares = (target_value / price_nonzero).fillna(0.0)

            trade_records.append({'date': dt, 'turnover': turnover, 'commission': float(cost)})

            if last_trade_nav is not None:
                period_return = (total_equity - last_trade_nav) / last_trade_nav
                period_pnls.append({'start': last_trade_date, 'end': dt, 'return': float(period_return)})
            last_trade_nav, last_trade_date = total_equity, dt

        port_val = float(np.nansum(shares.values * prices.values))
        positions_value[dt] = port_val
        equity_curve[dt] = cash + port_val

    if last_trade_nav is not None:
        final_nav = equity_curve.iloc[-1]
        period_return = (final_nav - last_trade_nav) / last_trade_nav
        period_pnls.append({'start': last_trade_date, 'end': equity_curve.index[-1], 'return': float(period_return)})

    daily_ret = equity_curve.pct_change().dropna()

    # 计算季度收益
    quarterly_returns = calculate_quarterly_returns(equity_curve)
    
    # 计算年度化指标
    ann_return, total_return, vol_ann, sharpe, max_dd, calm_ratio = calculate_annualized_stats(
        equity_curve, daily_ret
    )
    
    # 计算新的宁静比率
    serenity_ratio, _, cdar, ulcer_index = calculate_serenity_ratio_new(
        equity_curve, daily_ret, rf_rate=RF_ANNUAL
    )
    
    # 计算SPY的宁静比率作为对比
    spy_serenity_ratio = calculate_spy_serenity_ratio(daily_ret.index, rf_rate=RF_ANNUAL)
    
    # 计算交易统计
    num_trades, win_rate = calculate_trade_stats(period_pnls)

    return {
        'equity_curve': equity_curve,
        'positions_value': positions_value,
        'daily_return': daily_ret,
        'quarterly_returns': quarterly_returns,
        'trade_records': pd.DataFrame(trade_records),
        'period_returns': pd.DataFrame(period_pnls),
        'ann_return': ann_return,
        'total_return': total_return,
        'vol_ann': vol_ann,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calm_ratio': serenity_ratio,  # 使用新的宁静比率
        'cdar': cdar,
        'ulcer_index': ulcer_index,
        'spy_calm_ratio': spy_serenity_ratio,  # SPY的宁静比率
        'num_trades': num_trades,
        'win_rate': win_rate
    }


def calculate_quarterly_returns(equity_curve):
    """计算季度收益"""
    q_df = pd.DataFrame({'nav': equity_curve})
    q_df['Q'] = q_df.index.to_period('Q')
    q_end_nav = q_df.groupby('Q').tail(1)
    q_start_nav = q_df.groupby('Q').head(1)
    q_returns = []
    for yq, end_row in q_end_nav.groupby('Q'):
        end_nav = end_row['nav'].iloc[-1]
        start_nav = q_start_nav[q_start_nav['Q'] == yq]['nav'].iloc[0]
        q_returns.append({'Quarter': str(yq), 'Return': float(end_nav / start_nav - 1.0)})
    return pd.DataFrame(q_returns)


def calculate_annualized_stats(equity_curve, daily_ret):
    """计算年度化统计指标"""
    ann_factor = 252.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    trading_days = len(daily_ret)
    ann_return = (1.0 + total_return) ** (ann_factor / trading_days) - 1.0 if trading_days > 0 else np.nan
    vol_ann = daily_ret.std() * np.sqrt(ann_factor)
    sharpe = ((daily_ret - RF_ANNUAL / ann_factor).mean() / daily_ret.std()) * np.sqrt(ann_factor) if daily_ret.std() > 0 else np.nan
    
    # 计算最大回撤（取绝对值转为正数）
    roll_max = equity_curve.cummax()
    max_dd = abs((equity_curve / roll_max - 1.0).min())  # 取绝对值
    
    # 计算宁静比率 (Calm Ratio)
    # 公式: (年化收益 - 无风险收益) / (年化波动率 × 最大回撤)
    denominator = vol_ann * max_dd
    if denominator > 1e-10:  # 避免除以接近零的数
        calm_ratio = (ann_return - RF_ANNUAL) / denominator
        print(calm_ratio)
    else:
        calm_ratio = np.nan
        print(calm_ratio)
    
    return ann_return, total_return, vol_ann, sharpe, max_dd, calm_ratio


def calculate_trade_stats(period_pnls):
    """计算交易统计"""
    periods_df = pd.DataFrame(period_pnls)
    num_trades = len(period_pnls)
    win_rate = float((periods_df['return'] > 0).mean()) if not periods_df.empty else np.nan
    return num_trades, win_rate


def calc_alpha(strategy_daily_ret: pd.Series, rf_annual=RF_ANNUAL):
    """计算Alpha和Beta"""
    from data_loader import load_or_download, ensure_columns_ohlcv
    import os
    
    def load_spy_for_alpha():
        """加载SPY数据用于Alpha计算"""
        path = os.path.join("./data/20210701_20250920", "SPY.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = ensure_columns_ohlcv(df)
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date').sort_index()[['close']]
        else:
            from data_loader import download_yfinance
            spy = download_yfinance('SPY', '2021-07-01', '2025-09-20')
            spy.to_csv(path, index=False)
            spy['date'] = pd.to_datetime(spy['date'])
            return spy.set_index('date').sort_index()[['close']]
    
    spy_df = load_spy_for_alpha()
    df = pd.DataFrame(index=strategy_daily_ret.index)
    df['ret_s'] = strategy_daily_ret.values
    df['spy_close'] = spy_df['close'].reindex(df.index).ffill()
    df['spy_ret'] = df['spy_close'].pct_change()
    df = df.dropna()
    
    if df.empty or df['spy_ret'].std() == 0:
        return np.nan, np.nan
    
    rf_daily = rf_annual / 252.0
    xs = df['ret_s'] - rf_daily
    xm = df['spy_ret'] - rf_daily
    X = np.vstack([np.ones(len(xm)), xm.values]).T
    y = xs.values
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    alpha_daily, beta = beta_hat[0], beta_hat[1]
    return alpha_daily * 252.0, beta


def calculate_spy_serenity_ratio(dates_index, rf_rate=0.03):
    """计算SPY的宁静比率作为对比"""
    from data_loader import load_or_download, ensure_columns_ohlcv
    import os
    
    def load_spy_data():
        """加载SPY数据"""
        path = os.path.join("./data/20210701_20250920", "SPY.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = ensure_columns_ohlcv(df)
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date').sort_index()[['close']]
        else:
            from data_loader import download_yfinance
            spy = download_yfinance('SPY', '2021-07-01', '2025-09-20')
            spy.to_csv(path, index=False)
            spy['date'] = pd.to_datetime(spy['date'])
            return spy.set_index('date').sort_index()[['close']]
    
    try:
        spy_df = load_spy_data()
        
        # 对齐日期索引
        spy_df = spy_df.reindex(dates_index).ffill().dropna()
        
        if len(spy_df) < 2:
            return np.nan
        
        # 计算SPY的日收益率
        spy_daily_ret = spy_df['close'].pct_change().dropna()
        
        # 计算SPY的净值曲线（假设初始为1）
        spy_equity_curve = (1 + spy_daily_ret).cumprod()
        
        # 计算SPY的宁静比率
        spy_serenity_ratio, _, _, _ = calculate_serenity_ratio_new(
            spy_equity_curve, spy_daily_ret, rf_rate=rf_rate
        )
        
        return spy_serenity_ratio
        
    except Exception as e:
        print(f"计算SPY宁静比率时出错: {e}")
        return np.nan


def yearly_stats(equity: pd.Series):
    """计算年度统计"""
    df = equity.to_frame('nav').copy()
    df['year'] = df.index.year
    stats = []
    for y, grp in df.groupby('year'):
        if len(grp) < 2:
            continue
        ret = grp['nav'].iloc[-1] / grp['nav'].iloc[0] - 1.0
        ann = (1 + ret) ** (252 / len(grp)) - 1.0
        stats.append({'Year': int(y), 'AnnualizedReturn': float(ann), 'TotalReturn': float(ret)})
    
    stats_df = pd.DataFrame(stats)
    last_year = df['year'].iloc[-1]
    ytd_grp = df[df['year'] == last_year]
    ytd = ytd_grp['nav'].iloc[-1] / ytd_grp['nav'].iloc[0] - 1.0 if len(ytd_grp) > 1 else np.nan
    
    t12m_window = equity.last('365D')
    t12m_ret = t12m_window.iloc[-1] / t12m_window.iloc[0] - 1.0 if len(t12m_window) >= 2 else np.nan
    
    return stats_df, ytd, t12m_ret
