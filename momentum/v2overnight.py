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

def compute_rsi_score(
    rsi_series: pd.Series, 
    rsi_volatility: Optional[float] = None, 
    sector_factor: float = 0.0
) -> float:
    """
    Computes a granular RSI score with dynamic thresholds based on RSI volatility and market conditions.
    Bull/Bear market zones are derived from the sector factor.

    Args:
        rsi_series (pd.Series): Series of RSI values.
        rsi_volatility (float, optional): Pre-computed RSI volatility. If None, it is calculated within the function.
        sector_factor (float): A factor representing sector performance (-2 to 2 range). Positive for bullish, negative for bearish.

    Returns:
        float: A score between 0 and 1 based on RSI, volatility, and sector dynamics.
    """
    if rsi_series.empty:
        return 0.5  # Neutral score if no data

    # Compute RSI volatility if not provided
    if rsi_volatility is None:
        lookback_window = 14  # Standard RSI period
        if len(rsi_series) < lookback_window:
            rsi_volatility = 1.0  # Default to 1 if insufficient data
        else:
            rsi_volatility = rsi_series.rolling(window=lookback_window).std().iloc[-1]
            if pd.isna(rsi_volatility):
                rsi_volatility = 1.0

    # Adjust thresholds based on volatility and sector factor
    vol_adj = 1 + (rsi_volatility - 1) * 0.2  # Scale thresholds based on volatility
    sector_adj = 1 + (sector_factor * 0.2)    # Adjust thresholds for bull/bear markets

    # Determine market condition
    is_bull = sector_factor >= 0

    if is_bull:
        # Bull market thresholds
        oversold_threshold = 40.0 * vol_adj * sector_adj
        neutral_low = 50.0 * vol_adj * sector_adj
        overbought_threshold = 80.0 * (2 - vol_adj) / sector_adj
        extreme_overbought = 90.0 * (2 - vol_adj) / sector_adj
    else:
        # Bear market thresholds
        oversold_threshold = 20.0 * vol_adj * sector_adj
        neutral_low = 30.0 * vol_adj * sector_adj
        overbought_threshold = 55.0 * (2 - vol_adj) / sector_adj
        extreme_overbought = 65.0 * (2 - vol_adj) / sector_adj

    # Latest RSI value
    rsi_val = rsi_series.iloc[-1]

    # Granular scoring logic
    if rsi_val < oversold_threshold:
        score = 1.0  # Strong buy signal
    elif rsi_val < neutral_low:
        score = 0.8  # Moderate buy signal
    elif rsi_val < overbought_threshold:
        score = 0.5  # Neutral
    elif rsi_val < extreme_overbought:
        score = 0.3  # Moderate sell signal
    else:
        score = 0.0  # Strong sell signal

    return float(score)

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

def compute_adx_di_score(df: pd.DataFrame, period: int = 14) -> float:
    """
    Evaluates trend strength and direction using ADX and +DI/-DI.
    Higher scores indicate strong bullish trends, while lower scores indicate strong bearish trends.

    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        period (int): Lookback period for ADX and directional indicators.

    Returns:
        float: A score between 0.0 and 1.0 based on ADX, +DI, and -DI.
    """
    if len(df) < period + 1:
        return 0.5  # Neutral score if insufficient data

    # Calculate the directional movement
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = np.maximum.reduce([tr1, tr2, tr3])

    smoothed_tr = pd.Series(true_range).rolling(period).mean()
    smoothed_plus_dm = pd.Series(plus_dm).rolling(period).mean()
    smoothed_minus_dm = pd.Series(minus_dm).rolling(period).mean()

    # Calculate +DI and -DI
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100

    # Calculate ADX
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = (di_diff / di_sum) * 100
    adx = dx.rolling(period).mean()

    # Get the latest values for ADX, +DI, and -DI
    latest_adx = adx.iloc[-1] if not adx.empty else 0.0
    latest_plus_di = plus_di.iloc[-1] if not plus_di.empty else 0.0
    latest_minus_di = minus_di.iloc[-1] if not minus_di.empty else 0.0

    # Scoring logic
    if latest_adx < 20:
        # Weak trend
        return 0.5
    elif latest_plus_di > latest_minus_di:
        # Bullish trend
        score = 0.7 + 0.3 * (latest_adx / 100)  # Scale ADX contribution
    else:
        # Bearish trend
        score = 0.3 - 0.3 * (latest_adx / 100)

    return float(np.clip(score, 0.0, 1.0))

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
            'sector': 1.0, 'rvol': 1.0, 'adx_di': 1.0, 'crossover': 1.0
        }
    if len(df) < 30:
        return 0.5

    # Possibly use historical dict if you like to store series for MACD etc.
    rsi_series = compute_rsi(df["Close"], period=14)
    sector_score = compute_sector_factor(sector_df, signal_date, 5, compare_index_df=compare_index_df)
    rsi_score_val = compute_rsi_score(rsi_series, sector_factor=sector_score)
    rsi_std_score_val = compute_rsi_std_score(rsi_series, 14, 60, market_trend=sector_score)
    macd_score_val = compute_macd_score(df["Close"], None)

    atr_series = compute_atr(df[["High", "Low", "Close"]], period=14)
    atr_filter_val = compute_atr_filter_score(df, atr_series, None)
    adx_di_val = compute_adx_di_score(df)
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
        "adx_di": adx_di_val,
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
    buy_score_threshold=0.7,
    account_balance=50000.0,
    allocation_pct=0.07,
    stop_loss_multiplier=1.5,
    profit_target_multiplier=3.0,
    weights_dict: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Now iterates day by day with all tickers simultaneously.
    Preserves the same intraday vs. overnight logic, earnings checks,
    and partial close logic as before, but in a single pass over dates.

    Returns:
        List of trades in the same structure as before, one dictionary per trade.
    """
    # Remove old trade file if it exists
    if os.path.exists(TRADE_LOG_FILE):
        os.remove(TRADE_LOG_FILE)

    logger.info(f"Starting Backtest from {start.date()} to {end.date()} on tickers: {tickers}")
    initial_balance = account_balance

    # Download all ticker data at once
    try:
        raw_data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True
        )
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return []

    # Fetch sector data
    sector_data = fetch_sector_data(sector_etf, start, end)
    if sector_data.empty or "Close" not in sector_data.columns:
        sector_data = pd.DataFrame({"Close": [1.0]}, index=[start])
        logger.warning("No valid sector data. Sector performance set to neutral=0.0")

    # Fetch compare index data
    compare_index_data = None
    if compare_index_etf:
        compare_index_data = fetch_sector_data(compare_index_etf, start, end)

    # Build per-ticker dataframes
    df_dict = {}
    if len(tickers) == 1:
        # In case there's only 1 ticker, the columns won't be multi-index
        # so raw_data itself is that single ticker's DataFrame
        df_dict[tickers[0]] = raw_data.dropna()
    else:
        # Multi-index columns: top level = ticker, second level = OHLC
        for tk in tickers:
            if tk not in raw_data.columns.levels[0]:
                logger.warning(f"No data for ticker {tk}. Skipping.")
                continue
            df_tk = raw_data[tk].dropna()
            if not df_tk.empty:
                df_dict[tk] = df_tk

    # Pre-fetch all earnings dates
    earnings_dict = {}
    for tk in tickers:
        if tk not in df_dict:
            earnings_dict[tk] = set()
            continue
        try:
            yft = yf.Ticker(tk)
            earnings_df = yft.get_earnings_dates(limit=20)
            if earnings_df is not None and not earnings_df.empty:
                # Some returns have DatetimeIndex, others have 'Earnings Date' column
                if isinstance(earnings_df.index, pd.DatetimeIndex):
                    earnings_dates = set(earnings_df.index.normalize())
                elif 'Earnings Date' in earnings_df.columns:
                    earnings_dates = set(pd.to_datetime(earnings_df['Earnings Date']).dt.normalize())
                else:
                    earnings_dates = set()
            else:
                earnings_dates = set()
        except Exception as exc:
            logger.warning(f"Could not fetch earnings for {tk}. {exc}")
            earnings_dates = set()
        earnings_dict[tk] = earnings_dates

    # Collect all relevant trading dates from all tickers
    all_dates = set()
    for tk, df_tk in df_dict.items():
        if not df_tk.empty:
            for d in df_tk.index:
                # Only consider dates in [start, end]
                if start <= d <= end:
                    all_dates.add(d)
    all_dates = sorted(all_dates)

    # A dictionary to track open positions
    # None => no open position
    # Or a dict with entry info, stop, profit, etc.
    positions = {tk: None for tk in tickers}

    # If we decide on day X that we will close at next day open,
    # store that info here, keyed by ticker.
    # We'll handle it at the start of each day's loop.
    close_next_open = {tk: False for tk in tickers}

    trades_list: List[Dict[str, Any]] = []

    # Iterate over each date in chronological order
    for day_idx, day in enumerate(all_dates):
        next_day = all_dates[day_idx+1] if (day_idx + 1 < len(all_dates)) else None

        # 1) First handle any positions flagged to close at next day open.
        for tk in tickers:
            if close_next_open[tk] and tk in df_dict:
                df_tk = df_dict[tk]
                if day in df_tk.index and positions[tk] is not None:
                    # Close at day open
                    pos = positions[tk]
                    exit_price = float(df_tk.loc[day, "Open"])
                    exit_date = day
                    exit_action = "NextDay_ScoreDrop"  # matches original logic
                    trade_open = True

                    # Finalize the trade
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    account_balance += pnl
                    win_loss_flag = 1 if pnl > 0 else 0
                    trade_return = pnl / (pos["entry_price"] * pos["shares"]) if pos["shares"] > 0 else 0.0

                    # Log trade
                    rsi_static = pos["indicators"]["rsi"]
                    rsi_std_val = pos["indicators"]["rsi_std"]
                    macd_static = pos["indicators"]["macd"]
                    atr_fval = pos["indicators"]["atr_filter"]
                    sector_mapped = pos["indicators"]["sector"]
                    rvol_val = pos["indicators"]["rvol"]
                    adx_di_static = pos["indicators"]["adx_di"]
                    crossover_static = pos["indicators"]["crossover"]
                    held_overnights = pos["held_overnights"]

                    line = (
                        f"{tk},{pos['entry_date']},{exit_date},"
                        f"{pos['entry_price']:.2f},{exit_price:.2f},{pos['shares']},"
                        f"{pnl:.2f},{win_loss_flag},{trade_return:.5f},"
                        f"{rsi_static:.3f},{rsi_std_val:.3f},{macd_static:.3f},{atr_fval:.3f},"
                        f"{sector_mapped:.3f},{rvol_val:.3f},{adx_di_static:.3f},{crossover_static:.3f},"
                        f"{held_overnights}\n"
                    )

                    with open(TRADE_LOG_FILE, "a") as fh:
                        fh.write(line)

                    # Add to trades_list
                    trades_list.append({
                        "Ticker": tk,
                        "EntryDate": pos["entry_date"],
                        "ExitDate": exit_date,
                        "EntryPrice": pos["entry_price"],
                        "ExitPrice": exit_price,
                        "Shares": pos["shares"],
                        "PnL": pnl,
                        "WinLoss": win_loss_flag,
                        "ReturnPerc": trade_return,
                        "RSI": rsi_static,
                        "RSI_STD": rsi_std_val,
                        "MACD": macd_static,
                        "ATR_Filter": atr_fval,
                        "Sector": sector_mapped,
                        "RelativeVolume": rvol_val,
                        "ADX_DI": adx_di_static,
                        "Crossover": crossover_static,
                        "HeldOvernights": held_overnights
                    })

                    positions[tk] = None
                close_next_open[tk] = False  # Reset

        # 2) Now process normal daily checks for each ticker
        for tk in tickers:
            # Skip if we have no data for this ticker or day not in that df
            if tk not in df_dict:
                continue
            df_tk = df_dict[tk]
            if day not in df_tk.index:
                continue

            # If we don't even have 30 days of history up to this day, skip
            df_tk_up_to_day = df_tk.loc[:day]
            if len(df_tk_up_to_day) < 30:
                continue

            # Calculate the daily final score
            score_val = final_score_indicators(
                df_tk_up_to_day,
                sector_data,
                day,
                compare_index_data,
                volatility_threshold=volatility_threshold,
                weights_dict=weights_dict
            )

            # If there's an existing open position, check intraday price action
            if positions[tk] is not None:
                pos = positions[tk]
                day_open = float(df_tk.loc[day, "Open"])
                day_high = float(df_tk.loc[day, "High"])
                day_low = float(df_tk.loc[day, "Low"])
                day_close = float(df_tk.loc[day, "Close"])

                # Intraday stop or profit
                # Stop first
                if day_low <= pos["stop_price"]:
                    exit_price = pos["stop_price"]
                    exit_date = day
                    exit_action = "SL"
                    trade_open = False
                # Then profit
                elif day_high >= pos["profit_price"]:
                    exit_price = pos["profit_price"]
                    exit_date = day
                    exit_action = "TP"
                    trade_open = False
                else:
                    # Not triggered stop or target
                    # Intraday vs overnight logic
                    if pos["entry_score"] < 0.75:
                        # Was an intraday trade => close at day's close
                        exit_price = day_close
                        exit_date = day
                        exit_action = "Intraday_Close"
                        trade_open = False
                    else:
                        # Overnight candidate
                        # Check if next_day is earnings => close EOD
                        if next_day is None:
                            # Last day => close EOD
                            exit_price = day_close
                            exit_date = day
                            exit_action = "LastDay_Close"
                            trade_open = False
                        else:
                            # See if next day is earnings
                            if next_day in earnings_dict[tk]:
                                # close at today's close
                                exit_price = day_close
                                exit_date = day
                                exit_action = "Earnings_NextDay"
                                trade_open = False
                            else:
                                # Potential hold overnight => we wait.
                                # We'll re-check the score next day morning (via close_next_open if needed)
                                # For now, we remain open, increment held overnights
                                pos["held_overnights"] += 1
                                trade_open = True

                if not trade_open:
                    # Close the trade
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    account_balance += pnl
                    win_loss_flag = 1 if pnl > 0 else 0
                    trade_return = pnl / (pos["entry_price"] * pos["shares"]) if pos["shares"] > 0 else 0.0

                    rsi_static = pos["indicators"]["rsi"]
                    rsi_std_val = pos["indicators"]["rsi_std"]
                    macd_static = pos["indicators"]["macd"]
                    atr_fval = pos["indicators"]["atr_filter"]
                    sector_mapped = pos["indicators"]["sector"]
                    rvol_val = pos["indicators"]["rvol"]
                    adx_di_static = pos["indicators"]["adx_di"]
                    crossover_static = pos["indicators"]["crossover"]
                    held_overnights = pos["held_overnights"]

                    line = (
                        f"{tk},{pos['entry_date']},{exit_date},"
                        f"{pos['entry_price']:.2f},{exit_price:.2f},{pos['shares']},"
                        f"{pnl:.2f},{win_loss_flag},{trade_return:.5f},"
                        f"{rsi_static:.3f},{rsi_std_val:.3f},{macd_static:.3f},{atr_fval:.3f},"
                        f"{sector_mapped:.3f},{rvol_val:.3f},{adx_di_static:.3f},{crossover_static:.3f},"
                        f"{held_overnights}\n"
                    )
                    with open(TRADE_LOG_FILE, "a") as fh:
                        fh.write(line)

                    trades_list.append({
                        "Ticker": tk,
                        "EntryDate": pos["entry_date"],
                        "ExitDate": exit_date,
                        "EntryPrice": pos["entry_price"],
                        "ExitPrice": exit_price,
                        "Shares": pos["shares"],
                        "PnL": pnl,
                        "WinLoss": win_loss_flag,
                        "ReturnPerc": trade_return,
                        "RSI": rsi_static,
                        "RSI_STD": rsi_std_val,
                        "MACD": macd_static,
                        "ATR_Filter": atr_fval,
                        "Sector": sector_mapped,
                        "RelativeVolume": rvol_val,
                        "ADX_DI": adx_di_static,
                        "Crossover": crossover_static,
                        "HeldOvernights": held_overnights
                    })
                    positions[tk] = None

                else:
                    # We remained open (overnight). Do nothing special right now.
                    pass

            else:
                # No open position => consider opening a new trade
                if score_val >= buy_score_threshold:
                    # Buy at today's open
                    day_open = float(df_tk.loc[day, "Open"])
                    position_value = account_balance * allocation_pct
                    shares = int(position_value // day_open)
                    if shares <= 0:
                        continue

                    # Dynamic ATR
                    ds_mult, dp_mult = compute_dynamic_atr_multiplier(df_tk_up_to_day, stop_loss_multiplier, profit_target_multiplier)
                    atr_val = compute_atr(df_tk_up_to_day[["High", "Low", "Close"]], 14).iloc[-1]
                    stop_price = day_open - (atr_val * ds_mult)
                    profit_price = day_open + (atr_val * dp_mult)

                    # Gather indicator static values for logging
                    rsi_series_ = compute_rsi(df_tk_up_to_day["Close"], 14)
                    sector_val = compute_sector_factor(sector_data, day, 5, compare_index_data)
                    rsi_std_val = compute_rsi_std_score(rsi_series_, 14, 60, sector_val)
                    macd_static = compute_macd_score(df_tk_up_to_day["Close"])
                    atr_series2 = compute_atr(df_tk_up_to_day[["High","Low","Close"]], 14)
                    atr_fval = compute_atr_filter_score(df_tk_up_to_day, atr_series2)
                    rsi_final = compute_rsi_score(rsi_series_, sector_factor=sector_val)
                    sector_mapped = (sector_val + 2.0) / 4.0
                    rvol_val = compute_relative_volume_score(df_tk_up_to_day)
                    adx_di_static = compute_adx_di_score(df_tk_up_to_day)
                    sma10_ = df_tk_up_to_day["Close"].rolling(10).mean()
                    sma30_ = df_tk_up_to_day["Close"].rolling(30).mean()
                    diff_sma_ = sma10_ - sma30_ if len(sma10_) == len(sma30_) else None
                    crossover_static = compute_crossover_score(df_tk_up_to_day, 10, 30, diff_sma_)

                    positions[tk] = {
                        "entry_date": day,
                        "entry_price": day_open,
                        "shares": shares,
                        "stop_price": stop_price,
                        "profit_price": profit_price,
                        "entry_score": score_val,
                        "held_overnights": 0,
                        "indicators": {
                            "rsi": rsi_final,
                            "rsi_std": rsi_std_val,
                            "macd": macd_static,
                            "atr_filter": atr_fval,
                            "sector": sector_mapped,
                            "rvol": rvol_val,
                            "adx_di": adx_di_static,
                            "crossover": crossover_static
                        }
                    }

    # End of loop over all_dates
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
    # The 8 indicators: RSI, RSI_STD, MACD, ATR_Filter, Sector, Rvol, ADX_DI, Crossover
    names = ['rsi','rsi_std','macd','atr_filter','sector','rvol','adx_di','crossover']
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
    decay_rate: float = 0.005,
    min_lr: float = 1e-6
):
    """
    Iteratively optimize a trading strategy with dynamic learning rate adjustments,
    training the neural network on backtested trades, and updating indicator weights.
    
    Enhancements include:
      - Exponential decay of learning rate.
      - Sharpe-based plateau adjustments.
      - Trend and volatility analysis for robust updates.
      - Neural network training with trade outcomes.
      - Extraction and update of indicator weights from the model.
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
            'sector': 1.0, 'rvol': 1.0, 'adx_di': 1.0, 'crossover': 1.0
        }

    current_lr = initial_lr
    input_dim = len(base_weights)
    model = build_neural_model(input_dim, learning_rate=current_lr)
    scaler = StandardScaler()

    possible_dates = list(pd.date_range('2019-01-01', '2023-06-14', freq='D'))
    if not possible_dates:
        logger.error("No valid dates in range!")
        sys.exit(1)

    def summarize_trades(trades: List[Dict[str, Any]], initial_balance=50000.0):
        if not trades:
            return (0.0, 0.0, 0.0, 0.0)

        total_pnl = sum(t["PnL"] for t in trades)
        n_trades = len(trades)
        n_win = sum(t["WinLoss"] for t in trades)
        wr = n_win / n_trades if n_trades else 0.0
        rets = [t["ReturnPerc"] for t in trades]
        sr = (np.mean(rets) / np.std(rets) * np.sqrt(len(rets))) if len(rets) > 1 and np.std(rets) != 0 else 0.0
        mdd = compute_max_drawdown(trades, initial_balance)
        return (float(total_pnl), float(wr), float(sr), float(mdd))

    if os.path.exists(META_LOG_FILE):
        os.remove(META_LOG_FILE)

    def exponential_decay_lr(base_lr, iteration, decay_rate, min_lr):
        decayed = base_lr * np.exp(-decay_rate * iteration)
        return max(decayed, min_lr)

    sharpe_history = []

    # Define the feature order for training.
    # Note: The keys below match those used when logging trades.
    feature_order = ['rsi', 'rsi_std', 'macd', 'atr_filter', 'sector', 'rvol', 'adx_di', 'crossover']

    for itx in range(1, iterations + 1):
        logger.info(f"\n=== Iteration {itx} ===")

        # Select random training/validation date ranges.
        start_date = random.choice(possible_dates)
        train_start, train_end = start_date, start_date + pd.DateOffset(months=12)
        val_start, val_end = train_end + pd.Timedelta(days=1), train_end + pd.DateOffset(months=6)
        cutoff = pd.Timestamp("2025-01-14")
        train_end, val_end = min(train_end, cutoff), min(val_end, cutoff)

        if train_start >= train_end:
            logger.warning("Train window invalid, skipping iteration.")
            continue

        tickers_chosen = choose_valid_10_tickers_per_ticker(stock_lib, train_start, train_end, max_attempts=200)
        if len(tickers_chosen) < 10:
            logger.warning(f"Only {len(tickers_chosen)} valid tickers found after replacements. Proceeding.")

        # Run backtests on training data.
        train_trades = backtest_strategy(
            tickers_chosen, train_start, train_end, sector_etf="XLK", compare_index_etf="SPY",
            volatility_threshold=1.0, buy_score_threshold=0.7, account_balance=50000.0,
            allocation_pct=0.07, stop_loss_multiplier=1.5, profit_target_multiplier=3.0, weights_dict=base_weights
        )
        if not train_trades:
            logger.info("No trades or no valid data in training period => skip iteration.")
            with open(META_LOG_FILE, "a") as f:
                f.write(f"Iteration {itx}, TRAIN => No trades\n")
            continue

        total_pnl_train, wr_train, sr_train, dd_train = summarize_trades(train_trades, 50000.0)

        # Run validation backtest.
        val_trades = backtest_strategy(
            tickers_chosen, val_start, val_end, sector_etf="XLK", compare_index_etf="SPY",
            volatility_threshold=1.0, buy_score_threshold=0.7, account_balance=50000.0,
            allocation_pct=0.07, stop_loss_multiplier=1.5, profit_target_multiplier=3.0, weights_dict=base_weights
        ) if val_start < val_end else []

        total_pnl_val, wr_val, sr_val, dd_val = summarize_trades(val_trades, 50000.0)
        sharpe_history.append(sr_val)

        log_msg_train = (f"Iteration {itx}, ({train_start} to {train_end}) TRAIN => PnL={total_pnl_train:.2f}, "
                         f"WinRate={wr_train*100:.2f}%, Sharpe={sr_train:.3f}, Drawdown={dd_train*100:.2f}%\n")
        log_msg_val = (f"Iteration {itx}, ({val_start} to {val_end}) VALID => PnL={total_pnl_val:.2f}, "
                       f"WinRate={wr_val*100:.2f}%, Sharpe={sr_val:.3f}, Drawdown={dd_val*100:.2f}%\n")
        logger.info(log_msg_train.strip())
        logger.info(log_msg_val.strip())

        with open(META_LOG_FILE, "a") as f:
            f.write(log_msg_train)
            f.write(log_msg_val)

        # Adjust the learning rate.
        decayed_lr = exponential_decay_lr(initial_lr, itx, decay_rate, min_lr)
        combined_sharpe = (0.3 * sr_train + 0.7 * sr_val)

        if len(sharpe_history) > 5:
            trend = np.polyfit(range(5), sharpe_history[-5:], 1)[0]
            if trend > 0:
                decayed_lr = min(decayed_lr * 1.1, initial_lr)
            elif trend < 0:
                decayed_lr = max(decayed_lr * 0.7, min_lr)

        if combined_sharpe < 0.2:
            decayed_lr = max(decayed_lr * 0.5, min_lr)
        elif combined_sharpe > 1.0:
            decayed_lr = min(decayed_lr * 1.1, initial_lr)

        current_lr = decayed_lr
        if hasattr(model.optimizer, 'learning_rate'):
            model.optimizer.learning_rate.assign(current_lr)
        logger.info(f"Adjusted learning rate to {current_lr:.6f}")

        # ===== New: Build a training dataset from the training trades =====
        # For each trade, the feature vector is built based on the indicator values.
        # Expected keys in the trade dict:
        # "RSI", "RSI_STD", "MACD", "ATR_Filter", "Sector", "RelativeVolume", "ADX_DI", "Crossover"
        X_train = []
        y_train_pnl = []
        y_train_win = []
        for trade in train_trades:
            try:
                # Build features in the order defined by feature_order.
                row = [
                    trade["RSI"],
                    trade["RSI_STD"],
                    trade["MACD"],
                    trade["ATR_Filter"],
                    trade["Sector"],
                    trade["RelativeVolume"],
                    trade["ADX_DI"],
                    trade["Crossover"]
                ]
                X_train.append(row)
                y_train_pnl.append(trade["PnL"])
                y_train_win.append(trade["WinLoss"])
            except KeyError as e:
                logger.warning(f"Missing expected key {e} in trade data; skipping trade.")
        
        if X_train:
            X_train = np.array(X_train)
            y_train_pnl = np.array(y_train_pnl)
            y_train_win = np.array(y_train_win)

            # Scale the training features.
            X_train_scaled = scaler.fit_transform(X_train)

            # Train the model for a small number of epochs.
            # Adjust epochs and batch_size as needed based on your data.
            history = model.fit(
                X_train_scaled,
                {"pnl": y_train_pnl, "win_loss": y_train_win},
                epochs=5,
                verbose=0
            )
            logger.info(f"Training loss: {history.history}")
        else:
            logger.info("No valid training data extracted; skipping model update for this iteration.")

        # ===== Extract new indicator weights from the model =====
        new_weights = extract_indicator_weights(model)
        logger.info(f"Extracted new indicator weights: {new_weights}")
        base_weights = new_weights  # Update base weights for subsequent iterations

    return base_weights

###############################################################################
#                                   MAIN                                      #
###############################################################################
def main():
    
    logger.info("=== Starting Revised Neural Score Optimization ===")
    final_w = iterative_optimization(
        stock_library_csv="stock_library.csv",
        iterations=3,  # Adjust iteration count as desired
        base_weights={"rsi": 0.1, "rsi_std": 0.1, "macd": 0.1, "atr_filter": 0.1, "sector": 0.1, "rvol": 0.1, "adx_di": 0.1, "crossover": 0.1}
    )
    logger.info(f"Final Weights after all iterations: {final_w}")
    with open("optimized_weights.json", "w") as f:
        json.dump(final_w, f)
    logger.info("Saved final weights to optimized_weights.json.")

if __name__ == "__main__":
    main()
