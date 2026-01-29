# -*- coding: utf-8 -*-
"""
数据加载模块
负责数据获取、清洗和标准化
"""

import os
import pandas as pd
import numpy as np
from config import DATA_DIR


def csv_path(code: str) -> str:
    """获取CSV文件路径"""
    return os.path.join(DATA_DIR, f"{code}.csv")


def ensure_columns_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化OHLCV数据列
    - 支持中英文列名
    - 自动识别并重命名列
    - 数据清洗和格式标准化
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    
    variants = {
        'date': ['date', '日期', 'time', '交易日期'],
        'open': ['open', '开盘'],
        'high': ['high', '最高'],
        'low': ['low', '最低'],
        'close': ['close', '收盘', '收盘价', '收盘價', '收盘价(前复权)', 'adj close'],
        'volume': ['volume', '成交量', 'vol', '成交量(手)']
    }
    
    rename = {}
    for tgt, alts in variants.items():
        hit = None
        # 精确匹配
        for alt in alts:
            al = alt.lower()
            if al in lower_map:
                hit = lower_map[al]
                break
        # 包含匹配
        if hit is None:
            alts_l = [a.lower() for a in alts]
            for col in df.columns:
                cl = col.lower()
                if any(a in cl for a in alts_l):
                    hit = col
                    break
        if hit is not None:
            rename[hit] = tgt
    
    df = df.rename(columns=rename)
    
    # yfinance标准回退
    upper_fb = {'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'}
    for k, v in upper_fb.items():
        if v not in df.columns and k in df.columns:
            df = df.rename(columns={k: v})

    keep = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in keep if c in df.columns]].copy()
    
    if 'date' not in df.columns:
        raise ValueError("Missing 'date' after normalization.")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['date']).sort_values('date').drop_duplicates('date', keep='last')
    
    for k in ['open', 'high', 'low', 'close', 'volume']:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors='coerce')
    
    if 'close' not in df.columns:
        raise ValueError("Missing 'close' after normalization.")
    
    df = df.dropna(subset=['close'])
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


def download_akshare_etf(code: str, start: str, end: str) -> pd.DataFrame:
    """使用akshare下载ETF数据"""
    import akshare as ak
    try:
        df = ak.fund_etf_hist_em(
            symbol=code, period="daily",
            start_date=start.replace('-', ''),
            end_date=end.replace('-', ''),
            adjust=""
        )
    except Exception:
        df = ak.fund_etf_hist_sina(symbol=code)
    df = ensure_columns_ohlcv(df)
    m = (df['date'] >= start) & (df['date'] <= end)
    return df.loc[m].copy()


def download_yfinance(code: str, start: str, end: str) -> pd.DataFrame:
    """使用yfinance下载数据"""
    import yfinance as yf
    end_plus = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # inclusive end
    yfd = yf.download(code, start=start, end=end_plus, progress=False, auto_adjust=False)
    if yfd is None or yfd.empty:
        raise RuntimeError(f"yfinance empty for {code}")
    yfd = yfd.reset_index().rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    yfd['date'] = pd.to_datetime(yfd['date']).dt.tz_localize(None)
    yfd = yfd.sort_values('date')
    yfd['date'] = yfd['date'].dt.strftime('%Y-%m-%d')
    return yfd[['date', 'open', 'high', 'low', 'close', 'volume']]


def load_or_download(code: str, start: str, end: str) -> pd.DataFrame:
    """
    加载或下载数据
    - 优先使用缓存
    - 数字代码使用akshare，字母代码使用yfinance
    """
    path = csv_path(code)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = ensure_columns_ohlcv(df)
            m = (df['date'] >= start) & (df['date'] <= end)
            df = df.loc[m].copy()
            if not df.empty:
                return df
        except Exception:
            pass
    
    df = download_akshare_etf(code, start, end) if code.isdigit() else download_yfinance(code, start, end)
    df = ensure_columns_ohlcv(df)
    df.to_csv(path, index=False, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    return df


def get_all_price_series(codes, start, end):
    """获取所有资产的价格序列"""
    data = {}
    for code in codes:
        df = load_or_download(code, start, end)
        if df.empty:
            raise RuntimeError(f"No data for {code} in {start}~{end}")
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        data[code] = df[['open', 'high', 'low', 'close', 'volume']].copy()
    return data


def prepare_close_matrix(prices_dict, start_date, end_date):
    """准备收盘价矩阵"""
    all_trading_days = pd.Index(sorted(set().union(*[df.index for df in prices_dict.values()])))
    
    close_code = pd.DataFrame(index=all_trading_days, columns=list(prices_dict.keys()), dtype=float)
    for code, df in prices_dict.items():
        close_code.loc[df.index, code] = df['close']
    
    close_code = close_code.ffill().dropna(how='any')
    close_code = close_code.loc[
        (close_code.index >= pd.to_datetime(start_date)) &
        (close_code.index <= pd.to_datetime(end_date))
    ]
    
    return close_code
