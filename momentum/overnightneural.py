#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neuralscore.py

Combines a score-based multi-indicator approach with iterative backtesting and 
multi-output neural network training for momentum strategies.

Now updated to include:
 - Intraday vs. swing (overnight) logic
 - Earnings date checks (avoid overnight if next day is earnings)
 - Overnight performance monitoring
 - Validation feedback for learning rate adjustments
"""

import os
import sys
import json
import random
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# For the neural network
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

warnings.simplefilter("ignore", category=FutureWarning)

###############################################################################
#                          LOGGING CONFIGURATION                              #
###############################################################################
LOG_LEVEL = logging.INFO  # Adjust as needed
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("NeuralScoreLogger")

# Mute yfinance logger
logger_yf = logging.getLogger('yfinance')
logger_yf.disabled = True
logger_yf.propagate = False

TRADE_LOG_FILE = "trade_details.csv"  # CSV to which backtest trades (with indicators) are logged
META_LOG_FILE = "meta_log.txt"         # Text file for iteration-level summaries
OVERNIGHT_LOG_FILE = "overnight_trade_details.csv"

###############################################################################
#                        INDICATOR & HELPER FUNCTIONS                         #
###############################################################################
def compute_percentile_rank(series: pd.Series, value: float) -> float:
    if len(series) < 2:
        return 0.5
    return (series < value).mean()

def compute_macd(prices: pd.Series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_rsi(prices: pd.Series, period=14) -> pd.Series:
    if len(prices) < period + 1:
        return pd.Series([50]*len(prices), index=prices.index)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_rsi_score(rsi_series: pd.Series, rsi_volatility=1.0, sector_factor=0.0) -> float:
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

def compute_rsi_std_score(rsi_series: pd.Series, window=14, lookback=60, market_trend=0.0) -> float:
    if len(rsi_series) < window:
        return 0.5
    curr_std = rsi_series.rolling(window).std().iloc[-1]
    if pd.isna(curr_std):
        return 0.5
    hist_window = min(lookback, len(rsi_series) - window + 1)
    historical = rsi_series.rolling(window).std().iloc[-hist_window:]
    # Use .all() or .any() if checking boolean conditions
    pr = (historical < curr_std).mean()
    score = 1.0 - pr
    score = float(np.clip(score + 0.2 * market_trend, 0.0, 1.0))
    return score

def compute_macd_score(prices: pd.Series, historical_macd_hist: Optional[pd.Series] = None) -> float:
    macd_line, signal_line, hist = compute_macd(prices)
    if len(hist) == 0:
        return 0.5
    latest_hist = hist.iloc[-1]
    if len(macd_line) > 1:
        prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
        curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
        bullish_cross = (prev_diff < 0) and (curr_diff > 0)
        bearish_cross = (prev_diff > 0) and (curr_diff < 0)
    else:
        bullish_cross = bearish_cross = False
    base_score = 0.7 if latest_hist > 0 else 0.3
    if historical_macd_hist is not None and len(historical_macd_hist) > 10:
        mag_pct = compute_percentile_rank(historical_macd_hist.abs(), abs(latest_hist))
        base_score *= (0.5 + 0.5 * mag_pct)
    if bullish_cross:
        base_score += 0.2
    elif bearish_cross:
        base_score -= 0.2
    return float(np.clip(base_score, 0.0, 1.0))

def compute_atr(df: pd.DataFrame, period=14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_atr_filter_score(df: pd.DataFrame, atr_series: pd.Series, historical_atr: Optional[pd.Series] = None) -> float:
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
    if len(df) < max(ma_periods):
        return 0.5
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    ma200 = df["Close"].rolling(200).mean().iloc[-1]
    if (ma20 > ma50) and (ma50 > ma200):
        spacing = (ma20 - ma50) + (ma50 - ma200)
        rel_spacing = spacing / ma200 if ma200 != 0 else 0
        return float(np.clip(0.5 + rel_spacing, 0.0, 1.0))
    elif (ma20 < ma50) and (ma50 < ma200):
        return 0.0
    else:
        return 0.5

def compute_crossover_score(df: pd.DataFrame, short_period=10, long_period=30,
                            historical_diff: Optional[pd.Series] = None) -> float:
    if len(df) < long_period:
        return 0.5
    sma_short = df["Close"].rolling(short_period).mean().iloc[-1]
    sma_long = df["Close"].rolling(long_period).mean().iloc[-1]
    diff = sma_short - sma_long

    if historical_diff is not None and len(historical_diff) >= 2:
        velocity = historical_diff.iloc[-1] - historical_diff.iloc[-2]
        if (diff > 0) and (velocity > 0):
            return 1.0
        elif (diff < 0) and (velocity < 0):
            return 0.0
        else:
            return 0.5
    else:
        return 0.8 if diff > 0 else 0.2

def compute_relative_volume_score(df: pd.DataFrame,
                                  historical_rvol_distribution: Optional[pd.Series] = None,
                                  short_window=5, long_window=20) -> float:
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

def compute_sector_factor(sector_df: pd.DataFrame, signal_date: datetime, rolling_window=5,
                          compare_index_df: Optional[pd.DataFrame] = None) -> float:
    """Computes sector performance over a rolling window. Output in [-2,2] range."""
    if sector_df.empty or "Close" not in sector_df.columns:
        return 0.0
    # Ensure we find the largest date <= signal_date if not exact
    if signal_date not in sector_df.index:
        possible_dates = sector_df.index[sector_df.index <= signal_date]
        if possible_dates.empty:
            return 0.0
        signal_date = possible_dates[-1]

    pos = sector_df.index.get_loc(signal_date)
    start_pos = max(pos - rolling_window + 1, 0)
    segment = sector_df["Close"].iloc[start_pos:pos+1]

    val0 = segment.iloc[0]
    if hasattr(val0, '__len__') and not isinstance(val0, str):
        # Possibly array-like; check if all are zero
        if (val0 == 0).all():
            base_perf = 0.0
        else:
            base_perf = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]
    else:
        if val0 == 0:
            base_perf = 0.0
        else:
            base_perf = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]

    base_perf = float(base_perf)
    sector_score = np.clip(base_perf * 5, -1.0, 1.0)

    # Compare with index (SPY for ex) if provided
    if (compare_index_df is not None) and ("Close" in compare_index_df.columns):
        if signal_date not in compare_index_df.index:
            idx_dates = compare_index_df.index[compare_index_df.index <= signal_date]
            if idx_dates.empty:
                index_perf = 0.0
            else:
                sdate2 = idx_dates[-1]
                pos_idx = compare_index_df.index.get_loc(sdate2)
                start_idx = max(pos_idx - rolling_window + 1, 0)
                comp_segment = compare_index_df["Close"].iloc[start_idx:pos_idx+1]
                val_c0 = comp_segment.iloc[0]
                if hasattr(val_c0, '__len__') and not isinstance(val_c0, str):
                    if (val_c0 != 0).all():
                        index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]
                    else:
                        index_perf = 0.0
                else:
                    if val_c0 == 0:
                        index_perf = 0.0
                    else:
                        index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]
        else:
            pos_idx = compare_index_df.index.get_loc(signal_date)
            start_idx = max(pos_idx - rolling_window + 1, 0)
            comp_segment = compare_index_df["Close"].iloc[start_idx:pos_idx+1]
            val_c0 = comp_segment.iloc[0]
            if hasattr(val_c0, '__len__') and not isinstance(val_c0, str):
                if (val_c0 != 0).all():
                    index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]
                else:
                    index_perf = 0.0
            else:
                if val_c0 == 0:
                    index_perf = 0.0
                else:
                    index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]

        relative_perf = base_perf - index_perf
        sector_score += np.clip(relative_perf * 2, -0.5, 0.5)

    return float(np.clip(sector_score * 40, -2.0, 2.0))

def compute_sector_performance(sector_etf: str, start: datetime, end: datetime) -> float:
    df = yf.download(sector_etf, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     interval="1d", progress=False)
    if df.empty or "Close" not in df.columns:
        return 0.0
    first_price = df["Close"].iloc[0]
    last_price = df["Close"].iloc[-1]
    if hasattr(first_price, "__len__") and not isinstance(first_price, str):
        # Possibly array-like
        if (first_price != 0).all():
            return (last_price - first_price) / first_price
        else:
            return 0.0
    else:
        return (last_price - first_price) / first_price if first_price != 0 else 0.0

###############################################################################
#                          DRAWDOWN & EQUITY CURVE                            #
###############################################################################
def compute_max_drawdown(trades: List[Dict[str, Any]], initial_balance: float = 50000.0) -> float:
    """
    Compute maximum drawdown from a list of trades.
    We'll order trades by ExitDate and track equity after each exit.
    """
    if not trades:
        return 0.0

    # Convert exit dates to datetime, sort by exit date
    for t in trades:
        if not isinstance(t["ExitDate"], pd.Timestamp):
            t["ExitDate"] = pd.to_datetime(t["ExitDate"])

    trades_sorted = sorted(trades, key=lambda x: x["ExitDate"])
    equity = initial_balance
    peak = equity
    max_dd = 0.0
    for t in trades_sorted:
        equity += t["PnL"]
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak != 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd

###############################################################################
#                              FINAL SCORE FUNC                               #
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
    Combines these indicators into a final confidence score [0..1].
    We'll keep the weighting approach. 
    """
    if weights_dict is None:
        weights_dict = {
            'rsi': 1.0, 'rsi_std': 0.5, 'macd': 1.0, 'atr_filter': 1.0,
            'sector': 1.0, 'rvol': 1.0, 'multi_ma': 1.0, 'crossover': 1.0
        }
    if len(df) < 30:
        return 0.5

    # Possibly use historical dict if you like to store series for MACD etc.
    rsi_series = compute_rsi(df["Close"], period=14)
    rsi_vol = rsi_series.tail(14).std() / 10 if len(rsi_series) > 14 else 1.0
    sector_score = compute_sector_factor(sector_df, signal_date, 5, compare_index_df=compare_index_df)
    rsi_score_val = compute_rsi_score(rsi_series, rsi_volatility=rsi_vol, sector_factor=sector_score)
    market_trend = 1.0 if sector_score > 0 else 0.0
    rsi_std_score_val = compute_rsi_std_score(rsi_series, 14, 60, market_trend=market_trend)
    macd_score_val = compute_macd_score(df["Close"], None)

    atr_series = compute_atr(df[["High", "Low", "Close"]], period=14)
    atr_filter_val = compute_atr_filter_score(df, atr_series, None)
    multi_ma_val = compute_multi_ma_score(df)
    sma10 = df["Close"].rolling(10).mean()
    sma30 = df["Close"].rolling(30).mean()
    diff_sma_series = sma10 - sma30 if len(sma10) == len(sma30) else None
    crossover_score_val = compute_crossover_score(df, 10, 30, diff_sma_series)
    rvol_score_val = compute_relative_volume_score(df, None, 5, 20)

    # Map sector from [-2..2] to [0..1]
    sector_score_mapped = (sector_score + 2.0) / 4.0  

    indicator_scores = {
        "rsi": rsi_score_val,
        "rsi_std": rsi_std_score_val,
        "macd": macd_score_val,
        "atr_filter": atr_filter_val,
        "sector": sector_score_mapped,
        "rvol": rvol_score_val,
        "multi_ma": multi_ma_val,
        "crossover": crossover_score_val
    }

    weighted_sum = sum(indicator_scores[k] * weights_dict.get(k, 0.0) for k in indicator_scores)
    total_weight = sum(weights_dict.get(k, 0.0) for k in indicator_scores)
    if total_weight == 0:
        return 0.5
    raw_score = weighted_sum / total_weight
    penalty_factor = 1.0
    if atr_filter_val < 0.5:
        penalty_factor -= 0.10 * volatility_threshold
    final_score = raw_score * penalty_factor
    return float(np.clip(final_score, 0.0, 1.0))

def compute_dynamic_atr_multiplier(history_df: pd.DataFrame, base_stop_mult: float, base_profit_mult: float,
                                   period=14, lookback=60) -> (float, float):
    atr_series = compute_atr(history_df[["High", "Low", "Close"]], period)
    if atr_series.empty:
        return base_stop_mult, base_profit_mult
    current_atr = atr_series.iloc[-1]
    if len(atr_series) >= lookback:
        atr_median = atr_series.rolling(window=lookback).median().iloc[-1]
    else:
        atr_median = current_atr
    if pd.isna(atr_median) or atr_median <= 0:
        return base_stop_mult, base_profit_mult
    ratio = current_atr / atr_median
    ds_mult = base_stop_mult * ratio
    dp_mult = base_profit_mult * ratio
    # Ensure profit target always bigger than stop for clarity
    if dp_mult <= ds_mult:
        dp_mult = ds_mult + 1.0
    return ds_mult, dp_mult

def fetch_sector_data(etf_symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        df = yf.download(etf_symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                         interval="1d", progress=False)
        return df
    except Exception as e:
        logger.error(f"Error fetching sector data for {etf_symbol}: {e}")
        return pd.DataFrame()

###############################################################################
#                         BACKTEST (LOG ACTUAL INDICATORS)                    #
###############################################################################
def backtest_strategy(
    tickers: List[str],
    start: datetime,
    end: datetime,
    sector_etf="XLK",
    compare_index_etf: Optional[str] = "SPY",
    volatility_threshold=1.0,
    buy_score_threshold=0.65,
    account_balance=50000.0,
    allocation_pct=0.05,
    stop_loss_multiplier=1.5,
    profit_target_multiplier=3.0,
    weights_dict: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Now includes intraday vs. overnight logic:
      - If daily score in [buy_score_threshold, 0.75), treat as intraday (open & close same day).
      - If daily score >= 0.75, consider a swing trade (hold overnight) unless next day is earnings.
    Also logs an extra field "HeldOvernight" indicating how many nights (>=1) the position was held.
    Checks for next-day earnings before deciding to hold overnight.
    """

    # Remove old trade file if it exists
    if os.path.exists(TRADE_LOG_FILE):
        os.remove(TRADE_LOG_FILE)

    logger.info(f"Starting Backtest from {start.date()} to {end.date()} on tickers: {tickers}")
    initial_balance = account_balance

    # Download
    try:
        data = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                           interval="1d", group_by="ticker", progress=False, threads=True)
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return []

    # Sector + compare
    sector_data = fetch_sector_data(sector_etf, start, end)
    if sector_data.empty or "Close" not in sector_data.columns:
        sector_data = pd.DataFrame({"Close": [1.0]}, index=[start])
        logger.warning("No valid sector data. Sector performance set to neutral=0.0")
    compare_index_data = None
    if compare_index_etf:
        compare_index_data = fetch_sector_data(compare_index_etf, start, end)

    trades_list = []

    # NEW: Pre-fetch all earnings dates for each ticker once to avoid repeated calls
    earnings_dict = {}
    for tk in tickers:
        try:
            yft = yf.Ticker(tk)
            # Limit=20 for more coverage
            earnings_df = yft.get_earnings_dates(limit=20)
            if earnings_df is not None and not earnings_df.empty:
                # Convert index (DatetimeIndex) or 'Earnings Date' col to date set
                # yfinance may return a df with index=DatetimeIndex or a col with the date
                if isinstance(earnings_df.index, pd.DatetimeIndex):
                    # Some data is in the index
                    earnings_dates = set(earnings_df.index.normalize())
                else:
                    # Possibly in 'Earnings Date' col
                    if 'Earnings Date' in earnings_df.columns:
                        earnings_dates = set(pd.to_datetime(earnings_df['Earnings Date']).dt.normalize())
                    else:
                        earnings_dates = set()
            else:
                earnings_dates = set()
        except Exception as exc:
            logger.warning(f"Could not fetch earnings for {tk}. {exc}")
            earnings_dates = set()
        earnings_dict[tk] = earnings_dates

    for ticker in tickers:
        # Check data presence
        if ticker not in data.columns.levels[0]:
            logger.warning(f"No data for ticker {ticker}. Skipping.")
            continue

        df = data[ticker].dropna()
        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {ticker}. Need >=30 days. Skipping.")
            continue

        # We'll iterate day-by-day, but we allow multi-day holding for swing trades
        i = 30
        while i < len(df):
            row_date = df.index[i]
            history = df.iloc[:i+1]
            score_val = final_score_indicators(
                df.iloc[:i+1], sector_data, row_date, compare_index_data,
                volatility_threshold=volatility_threshold, weights_dict=weights_dict
            )

            if score_val >= buy_score_threshold:
                # Open position at today's Open
                entry_date = row_date
                entry_price = df["Open"].iloc[i]
                position_value = account_balance * allocation_pct
                shares = int(position_value // entry_price)
                if shares <= 0:
                    i += 1
                    continue

                # Dynamic ATR approach
                ds_mult, dp_mult = compute_dynamic_atr_multiplier(history, stop_loss_multiplier, profit_target_multiplier)
                atr_val = compute_atr(history[["High", "Low", "Close"]], 14).iloc[-1]
                stop_price = entry_price - (atr_val * ds_mult)
                profit_price = entry_price + (atr_val * dp_mult)

                # Now handle intraday vs. overnight
                # Intraday if score in [buy_score_threshold, 0.75)
                # Swing if score >= 0.75
                held_overnights = 0
                exit_date = row_date
                exit_price = entry_price
                exit_action = "EC"  # default "End-of-Candle" or "Close-of-Day"

                # We'll store the day we opened, then loop forward day by day until exit
                day_idx = i
                trade_open = True

                while trade_open and (day_idx < len(df)):
                    day_data = df.iloc[day_idx:day_idx+1]
                    current_date = day_data.index[0]
                    # Intraday check: if we immediately hit stop or profit
                    low_ = day_data["Low"].min()
                    high_ = day_data["High"].max()

                    if low_ <= stop_price:
                        exit_price = stop_price
                        exit_date = current_date
                        exit_action = "SL"
                        trade_open = False
                    elif high_ >= profit_price:
                        exit_price = profit_price
                        exit_date = current_date
                        exit_action = "TP"
                        trade_open = False
                    else:
                        # If still open at the day close
                        # Check if intraday trade => close at same day's close
                        if buy_score_threshold <= score_val < 0.75:
                            # Intraday: close at day's close
                            exit_price = day_data["Close"].iloc[-1]
                            exit_date = current_date
                            exit_action = "Intraday_Close"
                            trade_open = False
                        else:
                            # Swing: check if we want to hold overnight
                            # => we do if next day is not earnings & next day's score >= 0.75
                            # but we only can do that if there's a next day
                            if day_idx == (len(df) - 1):
                                # This is the last day, must exit now
                                exit_price = day_data["Close"].iloc[-1]
                                exit_date = current_date
                                exit_action = "LastDay_Close"
                                trade_open = False
                            else:
                                # Check if next day is earnings
                                next_day_date = df.index[day_idx+1].normalize()
                                if next_day_date in earnings_dict[ticker]:
                                    # We do NOT hold overnight; close at today's close
                                    exit_price = day_data["Close"].iloc[-1]
                                    exit_date = current_date
                                    exit_action = "Earnings_NextDay"
                                    trade_open = False
                                else:
                                    # We tentatively hold overnight, increment held_overnights
                                    # Next morning, re-check the score
                                    day_idx += 1
                                    if day_idx < len(df):
                                        # Evaluate next day’s open or next day’s new score
                                        new_date = df.index[day_idx]
                                        new_score = final_score_indicators(
                                            df.iloc[:day_idx+1],
                                            sector_data,
                                            new_date,
                                            compare_index_data,
                                            volatility_threshold=volatility_threshold,
                                            weights_dict=weights_dict
                                        )
                                        if new_score < 0.75:
                                            # We close at next day open
                                            exit_price = df["Open"].iloc[day_idx]
                                            exit_date = new_date
                                            exit_action = "NextDay_ScoreDrop"
                                            trade_open = False
                                        else:
                                            # Remain open, so overnight hold continues
                                            held_overnights += 1
                                    # Done handling next day check
                                    continue  # skip the day_idx increment at the bottom
                    # If we got here, the trade is closed or we break
                    day_idx += 1

                # Done with while loop => we either closed intraday or on some next day
                pnl = (exit_price - entry_price) * shares
                account_balance += pnl
                win_loss_flag = 1 if pnl > 0 else 0
                trade_return = pnl / (entry_price * shares) if shares > 0 else 0.0

                # Indicator values for logging
                rsi_series_ = compute_rsi(history["Close"], 14)
                rsi_static = rsi_series_.iloc[-1]
                rsi_std_val = compute_rsi_std_score(rsi_series_, 14, 60)
                macd_static = compute_macd_score(history["Close"])
                atr_series2 = compute_atr(history[["High","Low","Close"]], 14)
                atr_fval = compute_atr_filter_score(history, atr_series2)
                sector_val = compute_sector_factor(sector_data, row_date, 5, compare_index_data)
                sector_mapped = (sector_val + 2.0) / 4.0
                rvol_val = compute_relative_volume_score(history)
                multi_ma_static = compute_multi_ma_score(history)
                sma10_ = history["Close"].rolling(10).mean()
                sma30_ = history["Close"].rolling(30).mean()
                diff_sma_ = sma10_ - sma30_ if len(sma10_) == len(sma30_) else None
                crossover_static = compute_crossover_score(history, 10, 30, diff_sma_)

                line = (
                    f"{ticker},{entry_date},{exit_date},"
                    f"{entry_price:.2f},{exit_price:.2f},{shares},"
                    f"{pnl:.2f},{win_loss_flag},{trade_return:.5f},"
                    f"{rsi_static:.3f},{rsi_std_val:.3f},{macd_static:.3f},{atr_fval:.3f},"
                    f"{sector_mapped:.3f},{rvol_val:.3f},{multi_ma_static:.3f},{crossover_static:.3f},"
                    f"{held_overnights}\n"
                )
                if held_overnights > 0:
                    # Log overnight trades
                    with open(OVERNIGHT_LOG_FILE, "a") as fh:
                        fh.write(line)
                    with open(TRADE_LOG_FILE, "a") as fh:
                        fh.write(line)
                else:
                    # Log intraday trades
                    with open(TRADE_LOG_FILE, "a") as fh:
                        fh.write(line)

                trade_row = {
                    "Ticker": ticker,
                    "EntryDate": entry_date,
                    "ExitDate": exit_date,
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "Shares": shares,
                    "PnL": pnl,
                    "WinLoss": win_loss_flag,
                    "ReturnPerc": trade_return,
                    "RSI": rsi_static,
                    "RSI_STD": rsi_std_val,
                    "MACD": macd_static,
                    "ATR_Filter": atr_fval,
                    "Sector": sector_mapped,
                    "RelativeVolume": rvol_val,
                    "Multi_MA": multi_ma_static,
                    "Crossover": crossover_static,
                    "HeldOvernights": held_overnights
                }
                trades_list.append(trade_row)

                # Move i forward to day_idx (since we've consumed days up to exit day_idx)
                i = day_idx
            else:
                i += 1

    final_pnl = account_balance - initial_balance
    logger.info(f"Backtest completed. Start={initial_balance:.2f}, End={account_balance:.2f}, PnL={final_pnl:.2f}")
    return trades_list

###############################################################################
#                     PICKING EXACTLY 10 VALID TICKERS PER TICKER            #
###############################################################################
def pick_valid_ticker(stock_lib: pd.DataFrame, start_date: datetime, end_date: datetime) -> Optional[str]:
    """
    Picks one ticker at random, checks if it has >=30 days in [start_date, end_date]. 
    If valid, return ticker; else None.
    """
    tck = random.choice(stock_lib['Ticker'].tolist())
    try:
        df = yf.download(tck, 
                         start=start_date.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"),
                         interval="1d", progress=False)
        df = df.dropna()
        if len(df) >= 30:
            return tck
        else:
            return None
    except:
        return None

def choose_valid_10_tickers_per_ticker(stock_lib: pd.DataFrame, start_date: datetime, end_date: datetime,
                                       max_attempts=200) -> List[str]:
    """
    Picks exactly 10 tickers by individually checking each ticker's data.
    Each time we find an invalid ticker, we pick a new one. 
    Up to max_attempts overall.
    """
    valid_list = []
    attempts = 0
    while (len(valid_list) < 10) and (attempts < max_attempts):
        tck = pick_valid_ticker(stock_lib, start_date, end_date)
        attempts += 1
        if tck and (tck not in valid_list):
            valid_list.append(tck)
    if len(valid_list) < 10:
        logger.warning(f"Only {len(valid_list)} tickers found after {attempts} attempts.")
    return valid_list

###############################################################################
#                        NEURAL NETWORK (MULTI-OUTPUT)                        #
###############################################################################
def build_neural_model(input_dim: int, learning_rate=0.001) -> Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu', name="dense_shared")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)

    pnl_output = Dense(1, activation='linear', name='pnl')(x)
    win_loss_output = Dense(1, activation='sigmoid', name='win_loss')(x)

    model = Model(inputs=inputs, outputs=[pnl_output, win_loss_output])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'pnl': 'mse', 'win_loss': 'binary_crossentropy'},
        loss_weights={'pnl': 1.0, 'win_loss': 1.0},
        metrics={'pnl': 'mae', 'win_loss': 'accuracy'}
    )
    return model

def extract_indicator_weights(model: Model) -> dict:
    layer = model.get_layer("dense_shared")
    w = layer.get_weights()[0]  # shape: (input_dim, hidden_units)
    mean_w = np.mean(np.abs(w), axis=1)
    norm_w = mean_w / np.sum(mean_w)
    # The 8 indicators: RSI, RSI_STD, MACD, ATR_Filter, Sector, Rvol, Multi_MA, Crossover
    names = ['rsi','rsi_std','macd','atr_filter','sector','rvol','multi_ma','crossover']
    out = {}
    for i, nm in enumerate(names):
        out[nm] = float(norm_w[i]) if i < len(norm_w) else 0.0
    return out

###############################################################################
#                          ITERATIVE OPTIMIZATION                             #
###############################################################################
def iterative_optimization(
    stock_library_csv: str,
    iterations: int = 3,
    base_weights: Optional[Dict[str, float]] = None,
    initial_lr: float = 0.001,
    decay_rate: float = 0.01,
    min_lr: float = 1e-6
):
    """
    1) For each iteration:
       - Randomly choose a start_date for the 12-month training period
         and automatically define the subsequent 6-month validation period.
       - Attempt to pick EXACTLY 10 valid tickers, replacing invalid ones individually.
       - Backtest + log trades (TRAIN), parse CSV => X,y => train NN.
       - Backtest same 10 tickers on the VAL window => evaluate generalization.
       - Use validation performance to adjust the NN's learning rate for next iteration.
       - Log iteration-level results (PNL, WL, Sharpe, Drawdown) for both TRAIN and VAL to "meta_log.txt".

    Enhanced to:
       - Apply exponential decay to the learning rate as iterations progress.
       - Adjust LR further based on validation Sharpe ratio (plateau-based adjustment).
       - Ensure learning rate never goes below a specified minimum (min_lr).
    """
    try:
        stock_lib = pd.read_csv(stock_library_csv)
        if "Ticker" not in stock_lib.columns:
            logger.error("CSV must have 'Ticker' column.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading stock library: {e}")
        sys.exit(1)

    if not base_weights:
        base_weights = {
            'rsi': 1.0, 'rsi_std': 0.5, 'macd': 1.0, 'atr_filter': 1.0,
            'sector': 1.0, 'rvol': 1.0, 'multi_ma': 1.0, 'crossover': 1.0
        }

    # Build NN with some default LR (initialized to initial_lr).
    current_lr = initial_lr
    input_dim = len(base_weights)  # 8 indicators
    model = build_neural_model(input_dim, learning_rate=current_lr)
    scaler = StandardScaler()

    # Possible date range up to 2023-06-10 (ensuring we have room for training + validation).
    possible_dates = list(pd.date_range('2019-01-01','2023-06-10',freq='D'))
    if not possible_dates:
        logger.error("No valid dates in range!")
        sys.exit(1)

    # Helper to compute summary metrics.
    def summarize_trades(trades: List[Dict[str, Any]], initial_balance=50000.0):
        if not trades:
            return (0.0, 0.0, 0.0, 0.0)

        total_pnl = sum(t["PnL"] for t in trades)
        n_trades = len(trades)
        n_win = sum(t["WinLoss"] for t in trades)
        wr = n_win / n_trades if n_trades else 0.0
        rets = [t["ReturnPerc"] for t in trades]
        if len(rets) > 1:
            mean_r = np.mean(rets)
            std_r = np.std(rets)
            sr = mean_r / std_r * np.sqrt(len(rets)) if std_r != 0 else 0.0
        else:
            sr = 0.0
        mdd = compute_max_drawdown(trades, initial_balance)

        return (float(total_pnl), float(wr), float(sr), float(mdd))

    # Clear meta_log if exists.
    if os.path.exists(META_LOG_FILE):
        os.remove(META_LOG_FILE)

    # Define a small function for exponential decay.
    def exponential_decay_lr(base_lr, iteration, decay_rate, min_lr):
        """
        Returns an exponentially decayed learning rate: base_lr * exp(-decay_rate * iteration),
        with a floor at min_lr.
        """
        decayed = base_lr * np.exp(-decay_rate * iteration)
        return max(decayed, min_lr)

    # Log the initial learning rate.
    logger.info(f"Initial Learning Rate: {current_lr:.6f}")

    for itx in range(1, iterations + 1):
        logger.info(f"\n=== Iteration {itx} ===")

        # 1) pick random date for the 12-month training window
        start_date = random.choice(possible_dates)
        train_start = start_date
        train_end = start_date + pd.DateOffset(months=12)
        # validation period => next 6 months
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=6)

        # Clip end if beyond 2025-01-10
        cutoff = pd.Timestamp("2025-01-10")
        if train_end > cutoff:
            train_end = cutoff
        if val_end > cutoff:
            val_end = cutoff
        if train_start >= train_end:
            logger.warning("Train window invalid, skipping iteration.")
            continue

        # 2) pick EXACTLY 10 valid tickers from the training window
        tickers_chosen = choose_valid_10_tickers_per_ticker(stock_lib, train_start, train_end, max_attempts=200)
        if len(tickers_chosen) < 10:
            logger.warning(f"Only {len(tickers_chosen)} valid tickers found after replacements. We'll proceed.")

        # 3) Backtest => logs trades (TRAIN)
        train_trades = backtest_strategy(
            tickers_chosen, train_start, train_end,
            sector_etf="XLK", compare_index_etf="SPY",
            volatility_threshold=1.0, buy_score_threshold=0.65,
            account_balance=50000.0, allocation_pct=0.1,
            stop_loss_multiplier=1.5, profit_target_multiplier=3.0,
            weights_dict=base_weights
        )
        if not train_trades:
            logger.info("No trades or no valid data => skip iteration.")
            # Still log a line to meta_log
            with open(META_LOG_FILE, "a") as f:
                f.write(f"Iteration {itx}, TRAIN => No trades\n")
            continue

        # Summaries (TRAIN)
        total_pnl_train, wr_train, sr_train, dd_train = summarize_trades(train_trades, 50000.0)

        # 4) parse trade_details.csv => build X,y => train NN (TRAIN data)
        if not os.path.exists(TRADE_LOG_FILE):
            logger.warning("No trade log => skip training.")
        else:
            df_log = pd.read_csv(TRADE_LOG_FILE, header=None)
            if df_log.empty:
                logger.warning("Trade log empty => skip training.")
            else:
                df_log.columns = [
                    "Ticker","EntryDate","ExitDate","EntryPrice","ExitPrice","Shares",
                    "PnL","WinLoss","ReturnPerc",
                    "RSI","RSI_STD","MACD","ATR_Filter","Sector","RelativeVolume","Multi_MA","Crossover",
                    "HeldOvernights"
                ]
                X_cols = ["RSI","RSI_STD","MACD","ATR_Filter","Sector","RelativeVolume","Multi_MA","Crossover"]
                y_pnl = df_log["PnL"].values
                y_win = df_log["WinLoss"].values
                X_feat = df_log[X_cols].values  # shape: (N,8)

                if len(X_feat) >= 2:
                    X_scaled = scaler.fit_transform(X_feat)
                    model.fit(
                        X_scaled,
                        {"pnl": y_pnl, "win_loss": y_win},
                        epochs=5,
                        batch_size=16,
                        verbose=0
                    )
                    new_weights = extract_indicator_weights(model)
                    logger.info(f"Updated Weights: {new_weights}")
                    base_weights = new_weights

            # Remove train trade log so we can isolate validation trades
            if os.path.exists(TRADE_LOG_FILE):
                os.remove(TRADE_LOG_FILE)

        # 5) Now do a backtest for the VALIDATION window with the updated weights
        if val_start >= val_end:
            logger.warning("Validation window invalid => skip val backtest.")
            val_trades = []
        else:
            val_trades = backtest_strategy(
                tickers_chosen, val_start, val_end,
                sector_etf="XLK", compare_index_etf="SPY",
                volatility_threshold=1.0, buy_score_threshold=0.65,
                account_balance=50000.0, allocation_pct=0.1,
                stop_loss_multiplier=1.5, profit_target_multiplier=3.0,
                weights_dict=base_weights
            )

        total_pnl_val, wr_val, sr_val, dd_val = summarize_trades(val_trades, 50000.0)

        # 6) Log iteration results to meta_log
        log_msg_train = (
            f"Iteration {itx}, TRAIN => PnL={total_pnl_train:.2f}, "
            f"WinRate={wr_train*100:.2f}%, Sharpe={sr_train:.3f}, Drawdown={dd_train*100:.2f}%\n"
        )
        log_msg_val = (
            f"Iteration {itx}, VALID => PnL={total_pnl_val:.2f}, "
            f"WinRate={wr_val*100:.2f}%, Sharpe={sr_val:.3f}, Drawdown={dd_val*100:.2f}%\n"
        )
        logger.info(log_msg_train.strip())
        logger.info(log_msg_val.strip())

        with open(META_LOG_FILE, "a") as f:
            f.write(log_msg_train)
            f.write(log_msg_val)

        # === Enhanced Learning Rate Updates ===
        # 1) Exponential decay based on iteration
        decayed_lr = exponential_decay_lr(initial_lr, itx, decay_rate, min_lr)

        # 2) Plateau-based adjustment using Sharpe ratio
        #    For demonstration, we compare sr_val to thresholds.
        if sr_val < 0.2:
            # If performance is poor, cut LR in half, respecting min_lr
            decayed_lr = max(decayed_lr * 0.5, min_lr)
        elif sr_val > 1.0:
            # If performance is excellent, boost LR by 10%, but cap at initial_lr if desired
            decayed_lr = min(decayed_lr * 1.1, initial_lr)

        current_lr = decayed_lr

        logger.info(f"Adjusted learning rate to {current_lr:.6f} based on iteration={itx} and validation Sharpe={sr_val:.3f}")

        # Update model's LR for next iteration
        if hasattr(model.optimizer, 'learning_rate'):
            model.optimizer.learning_rate.assign(current_lr)

        # Remove val trade log for cleanliness
        if os.path.exists(TRADE_LOG_FILE):
            os.remove(TRADE_LOG_FILE)

    return base_weights

###############################################################################
#                                   MAIN                                      #
###############################################################################
def main():
    logger.info("=== Starting Revised Neural Score Optimization ===")
    final_w = iterative_optimization(
        stock_library_csv="stock_library.csv",
        iterations=300,  # Adjust iteration count as desired
        base_weights={"rsi": 0.14398610591888428, "rsi_std": 0.12112914770841599, "macd": 0.14132548868656158, "atr_filter": 0.10645662993192673, "sector": 0.11987239867448807, "rvol": 0.12684527039527893, "multi_ma": 0.12292256951332092, "crossover": 0.1174624040722847}
    )
    logger.info(f"Final Weights after all iterations: {final_w}")
    with open("optimized_weights.json", "w") as f:
        json.dump(final_w, f)
    logger.info("Saved final weights to optimized_weights.json.")

if __name__ == "__main__":
    main()
