#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scoretest.py

Refactored script incorporating adaptive, normalized, and percentile-based scoring
for multiple technical indicators.

Changes:
- Dynamic RSI Score with adjustable oversold/overbought thresholds.
- RSI_STD normalized using percentile ranking (with fallback) and robust NaN handling.
- MACD score based on histogram sign and crossover detection.
- ATR used as a filter with normalization relative to its historical range.
- Multi-MA score based on moving average spacing.
- SMA Crossover score that factors in crossover velocity.
- Relative Volume score based on percentile rank.
- Sector Performance computed via rolling averages (with optional index comparison).
- Final score computed using configurable weights.
- Trade logging now includes all indicator scores plus final score and PnL.
- Dynamic ATR multipliers are checked so that the profit target multiplier is always greater than the stop-loss multiplier.
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm  # For progress bar

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

###############################################################################
#                              LOGGING SETUP                                  #
###############################################################################
LOG_FILE = "log_scoretest.log"
TRADE_LOG_FILE = "trade_details.csv"  # Trade log output saved as CSV for easier future reference

from logging.handlers import RotatingFileHandler

# Setup main logger (for summary messages)
logger = logging.getLogger("MomentumLogger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Setup trade logger (for trade details; here we write CSV lines)
trade_logger = logging.getLogger("TradeLogger")
trade_logger.setLevel(logging.INFO)
trade_file_handler = RotatingFileHandler(TRADE_LOG_FILE, maxBytes=5_000_000, backupCount=5)
trade_formatter = logging.Formatter("%(message)s")
trade_file_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_file_handler)

###############################################################################
#                               HELPER FUNCTIONS                              #
###############################################################################
def compute_percentile_rank(series: pd.Series, value: float) -> float:
    """
    Returns the percentile rank of 'value' within the given 'series'.
    Range: [0.0, 1.0].
    """
    if len(series) < 2:
        return 0.5
    return (series < value).mean()

def normalize_value(value: float, lower_bound: float, upper_bound: float, clip_to_0_1: bool = True) -> float:
    """
    Normalizes 'value' to a 0-1 range given lower and upper bounds.
    """
    if upper_bound == lower_bound:
        return 0.5
    norm = (value - lower_bound) / (upper_bound - lower_bound)
    if clip_to_0_1:
        norm = np.clip(norm, 0.0, 1.0)
    return norm

###############################################################################
#                   ADDITIONAL INDICATOR: MACD FUNCTION                       #
###############################################################################
def compute_macd(prices: pd.Series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

###############################################################################
#                     INDICATOR & SCORING FUNCTIONS                           #
###############################################################################
def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Classic RSI calculation.
    """
    if len(prices) < period + 1:
        return pd.Series([50] * len(prices), index=prices.index)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_rsi_score(rsi_series: pd.Series, rsi_volatility: float = 1.0, sector_factor: float = 0.0) -> float:
    """
    Dynamic RSI scoring with thresholds adjusted by volatility and sector influence.
    """
    if rsi_series.empty:
        return 0.5
    rsi_val = rsi_series.iloc[-1]
    vol_adj = 1 + (rsi_volatility - 1) * 0.2  
    sector_adj = 1 + (sector_factor * 0.2)
    oversold_threshold = 40.0 * vol_adj * sector_adj
    overbought_threshold = 80.0 * (2 - vol_adj) / sector_adj
    if rsi_val < oversold_threshold:
        return 1.0 if rsi_val > (oversold_threshold / 2) else 1.2
    elif rsi_val > overbought_threshold:
        penalty = -0.5 if sector_factor > 0 else -1.0
        return penalty
    else:
        return 0.8

def compute_rsi_std_score(rsi_series: pd.Series, window: int = 14, lookback: int = 60, market_trend: float = 0.0) -> float:
    """
    Computes a percentile-based RSI_STD score.
    Lower RSI_STD (less volatility) yields a higher score.
    Additionally, if market_trend is positive (trending), penalizes high RSI_STD less.
    """
    if len(rsi_series) < window:
        return 0.5
    curr_std = rsi_series.rolling(window).std().iloc[-1]
    if pd.isna(curr_std):
        return 0.5
    hist_window = min(lookback, len(rsi_series) - window + 1)
    historical = rsi_series.rolling(window).std().iloc[-hist_window:]
    pr = (historical < curr_std).mean()
    score = 1.0 - pr
    score = np.clip(score + 0.2 * market_trend, 0.0, 1.0)
    return score

def compute_macd_score(prices: pd.Series, historical_macd_hist: Optional[pd.Series] = None) -> float:
    """
    Scores MACD based on the sign of the histogram and recent crossovers.
    """
    macd_line, signal_line, hist = compute_macd(prices)
    if len(hist) == 0:
        return 0.5
    latest_hist = hist.iloc[-1]
    if len(macd_line) > 1 and len(signal_line) > 1:
        prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
        curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
        bullish_cross = (prev_diff < 0 and curr_diff > 0)
        bearish_cross = (prev_diff > 0 and curr_diff < 0)
    else:
        bullish_cross = bearish_cross = False
    base_score = 0.7 if latest_hist > 0 else 0.3
    if historical_macd_hist is not None and len(historical_macd_hist) > 10:
        mag_pct = compute_percentile_rank(historical_macd_hist.abs(), abs(latest_hist))
        scale = 0.5 + 0.5 * mag_pct
        base_score *= scale
    if bullish_cross:
        base_score += 0.2
    elif bearish_cross:
        base_score -= 0.2
    return float(np.clip(base_score, 0.0, 1.0))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def compute_atr_filter_score(df: pd.DataFrame, atr_series: pd.Series, historical_atr: Optional[pd.Series] = None) -> float:
    """
    Uses ATR primarily as a filter: lower ATR relative to history yields a higher score.
    """
    if len(atr_series) == 0:
        return 0.5
    curr_atr = atr_series.iloc[-1]
    if historical_atr is not None and len(historical_atr) > 10:
        pct_rank = compute_percentile_rank(historical_atr, curr_atr)
        return 1.0 - pct_rank
    else:
        if curr_atr < 1.0:
            return 0.8
        elif curr_atr > 5.0:
            return 0.2
        else:
            return 0.5

def compute_multi_ma_score(df: pd.DataFrame, ma_periods=[20, 50, 200]) -> float:
    """
    Evaluates alignment of multiple SMAs. Bullish (short > mid > long) yields a higher score.
    """
    if len(df) < max(ma_periods):
        return 0.5
    ma_vals = {}
    for p in ma_periods:
        ma_vals[p] = df["Close"].rolling(p).mean().iloc[-1]
    sorted_periods = sorted(ma_periods)
    short_ma, mid_ma, long_ma = ma_vals[sorted_periods[0]], ma_vals[sorted_periods[1]], ma_vals[sorted_periods[2]]
    if short_ma > mid_ma > long_ma:
        spacing = (short_ma - mid_ma) + (mid_ma - long_ma)
        rel_spacing = spacing / long_ma
        return float(np.clip(0.5 + rel_spacing, 0.0, 1.0))
    elif short_ma < mid_ma < long_ma:
        return 0.0
    else:
        return 0.5

def compute_crossover_score(df: pd.DataFrame, short_period=10, long_period=30, historical_diff: Optional[pd.Series] = None) -> float:
    """
    Evaluates SMA10 vs. SMA30 crossover strength; using the relative difference.
    """
    if len(df) < long_period:
        return 0.5
    sma_short = df["Close"].rolling(short_period).mean().iloc[-1]
    sma_long = df["Close"].rolling(long_period).mean().iloc[-1]
    diff = sma_short - sma_long
    if historical_diff is not None and len(historical_diff) >= 2:
        velocity = historical_diff.iloc[-1] - historical_diff.iloc[-2]
        if diff > 0 and velocity > 0:
            return 1.0
        elif diff < 0 and velocity < 0:
            return 0.0
        else:
            return 0.5
    else:
        return 0.8 if diff > 0 else 0.2

def compute_relative_volume_score(df: pd.DataFrame, historical_rvol_distribution: Optional[pd.Series] = None, short_window=5, long_window=20) -> float:
    """
    Compares short-term and long-term average volume.
    """
    if len(df) < long_window:
        return 0.5
    short_avg = df["Volume"].tail(short_window).mean()
    long_avg = df["Volume"].tail(long_window).mean()
    if long_avg <= 0:
        return 0.5
    rvol = short_avg / long_avg
    if historical_rvol_distribution is not None and len(historical_rvol_distribution) > 10:
        return compute_percentile_rank(historical_rvol_distribution, rvol)
    else:
        if rvol >= 2.0:
            return 1.0
        elif rvol >= 1.5:
            return 0.7
        elif rvol >= 1.0:
            return 0.5
        else:
            return 0.3

def compute_sector_factor(sector_df: pd.DataFrame, signal_date: datetime, rolling_window: int = 5, compare_index_df: Optional[pd.DataFrame] = None) -> float:
    """
    Computes the sector performance over a rolling window ending at signal_date.
    If compare_index_df is provided, computes the relative outperformance and adjusts the sector score.
    Final output is scaled to the range [-2.0, 2.0].
    """
    if sector_df.empty or "Close" not in sector_df.columns:
        return 0.0
    if signal_date not in sector_df.index:
        possible_dates = sector_df.index[sector_df.index <= signal_date]
        if possible_dates.empty:
            return 0.0
        signal_date = possible_dates[-1]
    pos = sector_df.index.get_loc(signal_date)
    start_pos = max(pos - rolling_window + 1, 0)
    segment = sector_df["Close"].iloc[start_pos: pos+1]
    if (segment.iloc[0] == 0).all():
        base_perf = 0.0
    else:
        base_perf = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]
    base_perf = float(base_perf)
    sector_score = np.clip(base_perf * 5, -1.0, 1.0)
    if compare_index_df is not None and "Close" in compare_index_df.columns:
        if signal_date not in compare_index_df.index:
            possible_dates_idx = compare_index_df.index[compare_index_df.index <= signal_date]
            if possible_dates_idx.empty:
                index_perf = 0.0
            else:
                signal_date = possible_dates_idx[-1]
                pos_idx = compare_index_df.index.get_loc(signal_date)
                start_idx = max(pos_idx - rolling_window + 1, 0)
                comp_segment = compare_index_df["Close"].iloc[start_idx: pos_idx+1]
                if (comp_segment.iloc[0] == 0).all():
                    index_perf = 0.0
                else:
                    index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]
        else:
            pos_idx = compare_index_df.index.get_loc(signal_date)
            start_idx = max(pos_idx - rolling_window + 1, 0)
            comp_segment = compare_index_df["Close"].iloc[start_idx: pos_idx+1]
            if (comp_segment.iloc[0] == 0).all():
                index_perf = 0.0
            else:
                index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]
        relative_perf = base_perf - index_perf
        sector_score += np.clip(relative_perf * 2, -0.5, 0.5)
    return float(np.clip(sector_score * 40, -2.0, 2.0))

###############################################################################
#                  FINAL SCORE COMBINATION FUNCTION                           #
###############################################################################
def final_score_indicators(
    df: pd.DataFrame,
    sector_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    compare_index_df: Optional[pd.DataFrame] = None,
    historical_dict: Optional[Dict[str, pd.Series]] = None,
    volatility_threshold: float = 1.0,
    weights_dict: Optional[Dict[str, float]] = None
) -> float:
    """
    Combines indicator scores into a final confidence score in [0,1].
    Indicators: RSI, RSI_STD, MACD, ATR (as a filter), Sector, Relative Volume,
    Multi-MA, and Crossover.
    """
    if weights_dict is None:
        weights_dict = {
            'rsi': 1.0,
            'rsi_std': 0.5,
            'macd': 1.0,
            'atr_filter': 1.0,
            'sector': 1.0,
            'rvol': 1.0,
            'multi_ma': 1.0,
            'crossover': 1.0
        }
    if len(df) < 30:
        return 0.5

    hist_rsi_std = historical_dict.get('rsi_std', None) if historical_dict else None
    hist_macd = historical_dict.get('macd_hist', None) if historical_dict else None
    hist_atr = historical_dict.get('atr', None) if historical_dict else None
    hist_diff_sma = historical_dict.get('diff_sma10_sma30', None) if historical_dict else None
    hist_rvol = historical_dict.get('rvol', None) if historical_dict else None

    rsi_series = compute_rsi(df["Close"], period=14)
    rsi_vol = rsi_series.tail(14).std() / 10 if len(rsi_series) > 14 else 1.0

    # Compute sector score via the compare-index function
    sector_score = compute_sector_factor(sector_df, signal_date, rolling_window=5, compare_index_df=compare_index_df)

    bull_regime = True if sector_score >= 0 else False

    rsi_score_val = compute_rsi_score(rsi_series, rsi_volatility=rsi_vol, sector_factor=sector_score)
    # Use dynamic RSI_STD score; market trend is 1 if sector_score > 0, else 0.
    market_trend = 1.0 if sector_score > 0 else 0.0
    rsi_std_score_val = compute_rsi_std_score(rsi_series, window=14, lookback=60, market_trend=market_trend)
    macd_score_val = compute_macd_score(df["Close"], historical_macd_hist=hist_macd)
    atr_series = compute_atr(df[["High", "Low", "Close"]], period=14)
    atr_filter_val = compute_atr_filter_score(df, atr_series, historical_atr=hist_atr)
    multi_ma_val = compute_multi_ma_score(df)
    sma10 = df["Close"].rolling(10).mean()
    sma30 = df["Close"].rolling(30).mean()
    diff_sma_series = sma10 - sma30 if len(sma10) == len(sma30) else None
    crossover_score_val = compute_crossover_score(df, short_period=10, long_period=30, historical_diff=hist_diff_sma)
    rvol_score_val = compute_relative_volume_score(df, historical_rvol_distribution=hist_rvol, short_window=5, long_window=20)

    indicator_scores = {
        "rsi": rsi_score_val,
        "rsi_std": rsi_std_score_val,
        "macd": macd_score_val,
        "atr_filter": atr_filter_val,
        "sector": (sector_score + 1.0) / 2.0,  # Remapped from [-1,1] to [0,1]
        "rvol": rvol_score_val,
        "multi_ma": multi_ma_val,
        "crossover": crossover_score_val
    }
    
    # Remove daily indicator logging from here for brevity.
    weighted_sum = sum(indicator_scores[k] * weights_dict.get(k, 0) for k in indicator_scores)
    total_weight = sum(weights_dict.get(k, 0) for k in indicator_scores)
    if total_weight == 0:
        return 0.5
    raw_score = weighted_sum / total_weight

    penalty_factor = 1.0
    if atr_filter_val < 0.5:
        penalty_factor -= 0.1 * volatility_threshold

    final_score = raw_score * penalty_factor
    return float(np.clip(final_score, 0.0, 1.0))

###############################################################################
#                     DYNAMIC ATR MULTIPLIER FUNCTION                         #
###############################################################################
def compute_dynamic_atr_multiplier(history_df: pd.DataFrame, base_stop_mult: float, base_profit_mult: float,
                                   period: int = 14, lookback: int = 60) -> (float, float):
    atr_series = compute_atr(history_df[["High", "Low", "Close"]], period=period)
    current_atr = atr_series.iloc[-1]
    if len(atr_series) >= lookback:
        atr_median = atr_series.rolling(window=lookback).median().iloc[-1]
    else:
        atr_median = current_atr
    if atr_median > 0:
        ratio = current_atr / atr_median
        dynamic_stop_mult = base_stop_mult * ratio
        dynamic_profit_mult = base_profit_mult * ratio
        # Ensure profit target multiplier is strictly higher than stop loss multiplier.
        if dynamic_profit_mult <= dynamic_stop_mult:
            dynamic_profit_mult = dynamic_stop_mult + 1.0
    else:
        dynamic_stop_mult = base_stop_mult
        dynamic_profit_mult = base_profit_mult
    return dynamic_stop_mult, dynamic_profit_mult

###############################################################################
#                     SECTOR PERFORMANCE FUNCTION                             #
###############################################################################
def fetch_sector_data(etf_symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        df = yf.download(etf_symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                         interval="1d", progress=False)
        return df
    except Exception as e:
        logger.error(f"Error downloading sector data for {etf_symbol}: {e}")
        return pd.DataFrame()

###############################################################################
#  BACKTESTING FUNCTION WITH DYNAMIC STOP-LOSS, TRAILING, & DYNAMIC TARGET    #
###############################################################################
def backtest_strategy(
    tickers: List[str],
    start: datetime,
    end: datetime,
    sector_etf: str = "XLK",
    compare_index_etf: Optional[str] = None,
    volatility_threshold: float = 1.0,
    holding_period: int = 1,
    buy_score_threshold: float = 0.7,
    account_balance: float = 50000.0,
    allocation_pct: float = 0.05,
    stop_loss_multiplier: float = 1.5,
    profit_target_multiplier: float = 3.0,
    weights_dict: Optional[Dict[str, float]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Backtests the momentum strategy intraday using the multi-indicator scoring system.
    """
    logger.info(f"Starting Backtest from {start.date()} to {end.date()} on tickers: {tickers}")
    initial_balance = account_balance

    try:
        data = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                           interval="1d", group_by="ticker", progress=False, threads=True)
    except Exception as e:
        logger.error(f"Error downloading ticker data: {e}")
        sys.exit(1)

    sector_data = fetch_sector_data(sector_etf, start, end)
    if sector_data.empty or "Close" not in sector_data.columns:
        logger.warning(f"No data for ETF {sector_etf}. Sector performance set to 0.0.")
        sector_data = pd.DataFrame({"Close": [1.0]}, index=[start])

    compare_index_data = None
    if compare_index_etf:
        compare_index_data = fetch_sector_data(compare_index_etf, start, end)
        if compare_index_data.empty:
            logger.warning(f"No data for compare index {compare_index_etf}")

    trades = {ticker: [] for ticker in tickers}

    for ticker in tickers:
        if ticker not in data.columns.levels[0]:
            logger.warning(f"No data for ticker {ticker}.")
            continue
        df = data[ticker].dropna()
        if df.empty or len(df) < 30:
            logger.warning(f"Not enough data for {ticker}; require at least 30 days to compute indicators.")
            continue

        historical_dict = {}  # Optionally used for further enhancements

        for i in tqdm(range(30, len(df)), desc=f"Processing {ticker}", leave=False):
            signal_date = df.index[i]
            history = df.iloc[: i + 1]

            score_val = final_score_indicators(history, sector_df=sector_data, signal_date=signal_date,
                                                 compare_index_df=compare_index_data,
                                                 volatility_threshold=volatility_threshold,
                                                 weights_dict=weights_dict)

            if score_val >= buy_score_threshold:
                entry_date = signal_date
                entry_price = df["Open"].loc[entry_date]
                position_value = account_balance * allocation_pct
                shares = int(position_value // entry_price)
                if shares <= 0:
                    continue

                dynamic_stop_mult, dynamic_profit_mult = compute_dynamic_atr_multiplier(
                    history, stop_loss_multiplier, profit_target_multiplier, period=14, lookback=60
                )
                atr_val = compute_atr(history[["High", "Low", "Close"]], period=14).iloc[-1]
                stop_price = entry_price - (atr_val * dynamic_stop_mult)
                profit_target = entry_price + (atr_val * dynamic_profit_mult)

                # Gather individual indicator scores for trade logging.
                rsi_static = compute_rsi(history["Close"], period=14).iloc[-1]
                rsi_std_val = compute_rsi_std_score(compute_rsi(history["Close"], period=14), 14, 60)
                crossover_score_val = compute_crossover_score(history)
                ma_score_val = compute_multi_ma_score(history)
                macd_score_val = compute_macd_score(history["Close"])
                # For brevity, we log only the key indicator values.
                trade_details = {
                    "RSI": round(rsi_static, 2),
                    "RSI_STD": round(rsi_std_val, 2),
                    "MACD": round(macd_score_val, 2),
                    "ATR_Filter": round(compute_atr_filter_score(history, compute_atr(history[["High", "Low", "Close"]], period=14)), 2),
                    "Sector": round(compute_sector_factor(sector_data, signal_date, rolling_window=5, compare_index_df=compare_index_data), 2),
                    "Relative_Volume": round(compute_relative_volume_score(history), 2),
                    "Multi_MA": round(ma_score_val, 2),
                    "Crossover": round(crossover_score_val, 2)
                }

                trade_entry = {
                    "Date": entry_date,
                    "Action": "BUY",
                    "Price": float(entry_price),
                    "Shares": shares,
                    "Final_Score": round(score_val, 2),
                    "StopLoss": float(stop_price),
                    "ProfitTarget": float(profit_target),
                    "Indicators": trade_details
                }
                trades[ticker].append(trade_entry)

                day_data = df.loc[entry_date:entry_date]
                if day_data.empty:
                    continue

                if day_data["Low"].min() <= stop_price:
                    exit_price = stop_price
                    exit_action = "SL"
                    exit_date = day_data.index[day_data["Low"] <= stop_price][0]
                elif day_data["High"].max() >= profit_target:
                    exit_price = profit_target
                    exit_action = "TP"
                    exit_date = day_data.index[day_data["High"] >= profit_target][0]
                else:
                    exit_price = day_data["Close"].iloc[-1]
                    exit_action = "EC"
                    exit_date = day_data.index[-1]

                trade_exit = {
                    "Date": exit_date,
                    "Action": "SELL",
                    "Price": float(exit_price),
                    "Shares": shares,
                    "Final_Score": round(score_val, 2)
                }
                trades[ticker].append(trade_exit)
                pnl = (exit_price - entry_price) * shares

                # Compose a CSV-formatted log line with key trade details.
                trade_log_line = (
                    f"{ticker},"
                    f"{entry_date.strftime('%Y%m%d')},"
                    f"{exit_date.strftime('%Y%m%d')},"
                    f"{entry_price:.2f},"
                    f"{exit_price:.2f},"
                    f"{shares},"
                    f"{score_val:.2f},"
                    f"{pnl:.2f},"
                    f"{trade_details['RSI']},"
                    f"{trade_details['RSI_STD']},"
                    f"{trade_details['MACD']},"
                    f"{trade_details['ATR_Filter']},"
                    f"{trade_details['Sector']},"
                    f"{trade_details['Relative_Volume']},"
                    f"{trade_details['Multi_MA']},"
                    f"{trade_details['Crossover']}"
                )
                trade_logger.info(trade_log_line)

                account_balance += pnl
                next_day = entry_date + timedelta(days=1)
                i = np.searchsorted(df.index, next_day)

    total_pnl = account_balance - initial_balance
    logger.info(
        f"Backtest completed. Starting balance: ${initial_balance:.2f}, "
        f"Ending balance: ${account_balance:.2f}, Total PnL: ${total_pnl:.2f}"
    )
    return trades

###############################################################################
#                                 MAIN                                        #
###############################################################################
def main(backtest_mode: bool = True):
    logger.info("=== Starting Momentum Strategy (Refactored) ===")

    selected_tickers = [
        "BMTX", "GOAI", "KOPN", "MSAI", "QRVO", "SPSC", "BMI", "MARA",
        "ZENV", "BKKT", "WKEY", "MAPS", "MRAM", "ALAR", "AEXAF"
    ]

    if backtest_mode:
        logger.info("Running BACKTEST mode...")
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 1, 1)
        my_weights = {"rsi": 0.14778868854045868, "rsi_std": 0.11519519984722137, "macd": 0.1855209320783615, "atr_filter": 0.10873917490243912, "sector": 0.1176842674612999, "rvol": 0.10129396617412567, "multi_ma": 0.0903923287987709, "crossover": 0.13338537514209747}
        trades = backtest_strategy(
            selected_tickers,
            start_date,
            end_date,
            sector_etf="XLK",
            compare_index_etf="SPY",
            volatility_threshold=1.0,
            buy_score_threshold=0.7,
            account_balance=50000.0,
            allocation_pct=0.05,
            stop_loss_multiplier=1.5,
            profit_target_multiplier=3.0,
            weights_dict=my_weights
        )
        logger.info("=== Momentum Strategy Backtest Completed ===")
    else:
        logger.info("LIVE/PAPER mode not fully implemented in this example.")
        # Implement live/paper trading logic here.

if __name__ == "__main__":
    main(backtest_mode=True)
