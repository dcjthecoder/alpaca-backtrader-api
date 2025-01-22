#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intraday_neuralscore.py

Enhanced version with:
 - Extended NN training with more epochs & dynamic LR.
 - Sector data smoothing & fallback to daily.
 - Time zone fixes for tz-naive vs. tz-aware dataframes.
"""
import os
import sys
import json
import random
import logging
import warnings
import pytz
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

def compute_macd(prices: pd.Series, fast_period=6, slow_period=13, signal_period=5):
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_rsi(prices: pd.Series, period=7) -> pd.Series:
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
    For intraday data the standard period is reduced to 7.
    """
    if rsi_series.empty:
        return 0.5  # Neutral if no data

    # Use a 7-period lookback for intraday RSI volatility
    if rsi_volatility is None:
        lookback_window = 7
        if len(rsi_series) < lookback_window:
            rsi_volatility = 1.0
        else:
            rsi_volatility = rsi_series.rolling(window=lookback_window).std().iloc[-1]
            if pd.isna(rsi_volatility):
                rsi_volatility = 1.0

    vol_adj = 1 + (rsi_volatility - 1) * 0.2
    sector_adj = 1 + (sector_factor * 0.2)
    is_bull = sector_factor >= 0

    if is_bull:
        oversold_threshold = 40.0 * vol_adj * sector_adj
        neutral_low = 50.0 * vol_adj * sector_adj
        overbought_threshold = 80.0 * (2 - vol_adj) / sector_adj
        extreme_overbought = 90.0 * (2 - vol_adj) / sector_adj
    else:
        oversold_threshold = 20.0 * vol_adj * sector_adj
        neutral_low = 30.0 * vol_adj * sector_adj
        overbought_threshold = 55.0 * (2 - vol_adj) / sector_adj
        extreme_overbought = 65.0 * (2 - vol_adj) / sector_adj

    rsi_val = rsi_series.iloc[-1]
    if rsi_val < oversold_threshold:
        score = 1.0
    elif rsi_val < neutral_low:
        score = 0.8
    elif rsi_val < overbought_threshold:
        score = 0.5
    elif rsi_val < extreme_overbought:
        score = 0.3
    else:
        score = 0.0

    return float(score)

def compute_rsi_std_score(rsi_series: pd.Series, window=7, lookback=30, market_trend=0.0) -> float:
    if len(rsi_series) < window:
        return 0.5
    curr_std = rsi_series.rolling(window).std().iloc[-1]
    if pd.isna(curr_std):
        return 0.5
    hist_window = min(lookback, len(rsi_series) - window + 1)
    historical = rsi_series.rolling(window).std().iloc[-hist_window:]
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

def compute_atr(df: pd.DataFrame, period=10) -> pd.Series:
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

def compute_adx_di_score(df: pd.DataFrame, period: int = 10) -> float:
    if len(df) < period + 1:
        return 0.5

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

    plus_di = (smoothed_plus_dm / smoothed_tr) * 100
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100

    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = (di_diff / di_sum) * 100
    adx = dx.rolling(period).mean()

    latest_adx = adx.iloc[-1] if not adx.empty else 0.0
    latest_plus_di = plus_di.iloc[-1] if not plus_di.empty else 0.0
    latest_minus_di = minus_di.iloc[-1] if not minus_di.empty else 0.0

    if latest_adx < 20:
        return 0.5
    elif latest_plus_di > latest_minus_di:
        score = 0.7 + 0.3 * (latest_adx / 100)
    else:
        score = 0.3 - 0.3 * (latest_adx / 100)
    return float(np.clip(score, 0.0, 1.0))

def compute_crossover_score(df: pd.DataFrame, short_period=5, long_period=15,
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
    # Ensure that signal_date is in UTC:
    if signal_date.tzinfo is None:
        signal_date = signal_date.replace(tzinfo=pytz.UTC)
    else:
        signal_date = signal_date.astimezone(pytz.UTC)
    
    # Ensure we find the largest date <= signal_date.
    possible_dates = sector_df.index[sector_df.index <= signal_date]
    if possible_dates.empty:
        return 0.0
    signal_date = possible_dates[-1]

    pos = sector_df.index.get_loc(signal_date)
    start_pos = max(pos - rolling_window + 1, 0)
    segment = sector_df["Close"].iloc[start_pos:pos+1]

    val0 = segment.iloc[0]
    if hasattr(val0, '__len__') and not isinstance(val0, str):
        if (val0 == 0).all():
            base_perf = 0.0
        else:
            base_perf = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]
    else:
        base_perf = 0.0 if val0 == 0 else (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]

    base_perf = float(base_perf)
    sector_score = np.clip(base_perf * 5, -1.0, 1.0)

    if (compare_index_df is not None) and ("Close" in compare_index_df.columns):
        possible_dates_idx = compare_index_df.index[compare_index_df.index <= signal_date]
        if possible_dates_idx.empty:
            index_perf = 0.0
        else:
            sdate2 = possible_dates_idx[-1]
            pos_idx = compare_index_df.index.get_loc(sdate2)
            start_idx = max(pos_idx - rolling_window + 1, 0)
            comp_segment = compare_index_df["Close"].iloc[start_idx:pos_idx+1]
            val_c0 = comp_segment.iloc[0]
            if hasattr(val_c0, '__len__') and not isinstance(val_c0, str):
                index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0] if (val_c0 != 0).all() else 0.0
            else:
                index_perf = 0.0 if val_c0 == 0 else (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]
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
        if (first_price != 0).all():
            return (last_price - first_price) / first_price
        else:
            return 0.0
    else:
        return (last_price - first_price) / first_price if first_price != 0 else 0.0

def compute_max_drawdown(trades: List[Dict[str, Any]], initial_balance: float = 5000.0) -> float:
    if not trades:
        return 0.0
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

def final_score_indicators(
    df: pd.DataFrame,
    sector_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    compare_index_df: Optional[pd.DataFrame] = None,
    historical_dict: Optional[Dict[str, pd.Series]] = None,
    volatility_threshold: float = 1.0,
    weights_dict: Optional[Dict[str, float]] = None
) -> float:
    if weights_dict is None:
        weights_dict = {
            'rsi': 1.0, 'rsi_std': 0.5, 'macd': 1.0, 'atr_filter': 1.0,
            'sector': 1.0, 'rvol': 1.0, 'adx_di': 1.0, 'crossover': 1.0
        }
    if len(df) < 30:
        return 0.5

    # Use a 7-day window for intraday calculations.
    rsi_series = compute_rsi(df["Close"], period=7)
    sector_score = compute_sector_factor(sector_df, signal_date, 5, compare_index_df=compare_index_df)
    rsi_score_val = compute_rsi_score(rsi_series, sector_factor=sector_score)
    rsi_std_score_val = compute_rsi_std_score(rsi_series, 7, 30, market_trend=sector_score)
    macd_score_val = compute_macd_score(df["Close"], None)

    atr_series = compute_atr(df[["High", "Low", "Close"]], period=10)
    atr_filter_val = compute_atr_filter_score(df, atr_series, None)
    adx_di_val = compute_adx_di_score(df)
    sma_short = df["Close"].rolling(5).mean()
    sma_long = df["Close"].rolling(15).mean()
    diff_sma_series = sma_short - sma_long if len(sma_short) == len(sma_long) else None
    crossover_score_val = compute_crossover_score(df, 5, 15, diff_sma_series)
    rvol_score_val = compute_relative_volume_score(df, None, 5, 20)

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

    logger.debug(f"Final scores: {indicator_scores}")

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
                                   period=10, lookback=30) -> (float, float):
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
    if dp_mult <= ds_mult:
        dp_mult = ds_mult + 1.0
    return ds_mult, dp_mult

def fetch_sector_data(etf_symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        df = yf.download(
            etf_symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1h",  # Changed from "1d" to "1h" if needed, else keep "1d"
            progress=False
        )
        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
        return df
    except Exception as e:
        logger.error(f"Error fetching sector data for {etf_symbol}: {e}")
        return pd.DataFrame()
    
###############################################################################
#                     INTRADAY BACKTEST (OPEN/CLOSE SAME DAY)                 #
###############################################################################
def backtest_intraday_strategy(
    tickers: List[str],
    start: datetime,
    end: datetime,
    sector_etf: str = "QQQ",
    compare_index_etf: Optional[str] = "SPY",
    volatility_threshold: float = 1.0,
    buy_score_threshold: float = 0.7,
    account_balance: float = 5000.0,
    allocation_pct: float = 0.07,
    stop_loss_multiplier: float = 1.5,
    profit_target_multiplier: float = 3.0,
    weights_dict: Optional[Dict[str, float]] = None,
    PDT: bool = True
) -> List[Dict[str, Any]]:
    """
    Intraday backtest on 1-hour bars with PDT logic and multi‑day (overnight) checks.
    Features:
      - Interval="1h"
      - Timezone-aware processing (converted to UTC)
      - PDT logic: limits to 3 day trades per rolling 5‑day window.
      - Overnight logic: if a position is held for >72 hours, adjust stop loss to break‑even and force an exit.
    """
    # 1) Convert start/end to UTC
    start = start.astimezone(pytz.UTC)
    end = end.astimezone(pytz.UTC)

    # Remove old trade log file if present
    if os.path.exists(TRADE_LOG_FILE):
        os.remove(TRADE_LOG_FILE)

    logger.info(f"Starting Intraday (1h) Backtest from {start} to {end} on {tickers}")
    initial_balance = account_balance

    # 2) Download intraday 1h data
    try:
        raw_data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1h",  # Changed from "30m" to "1h"
            group_by="ticker",
            progress=False,
            threads=True
        )
    except Exception as e:
        logger.error(f"Error downloading intraday data: {e}")
        return []

    # 3) Ensure the DataFrame indices are timezone-aware (in UTC)
    df_dict = {}
    if len(tickers) == 1:
        df_single = raw_data.dropna()
        if not df_single.empty:
            if df_single.index.tz is None:
                df_single.index = df_single.index.tz_localize('UTC')
            else:
                df_single.index = df_single.index.tz_convert('UTC')
        df_dict[tickers[0]] = df_single
    else:
        for tk in tickers:
            if tk in raw_data.columns.levels[0]:
                df_tk = raw_data[tk].dropna()
                if not df_tk.empty:
                    if df_tk.index.tz is None:
                        df_tk.index = df_tk.index.tz_localize('UTC')
                    else:
                        df_tk.index = df_tk.index.tz_convert('UTC')
                    df_dict[tk] = df_tk

    # 4) Sector & Compare Data
    sector_data = fetch_sector_data(sector_etf, start, end)
    if sector_data.empty or "Close" not in sector_data.columns:
        sector_data = pd.DataFrame({"Close": [1.0]}, index=[start])
        logger.warning("No valid sector data => default sector=0.0")
    compare_index_data = None
    if compare_index_etf:
        compare_index_data = fetch_sector_data(compare_index_etf, start, end)

    # 5) Pre-fetch earnings dates for each ticker
    earnings_dict = {}
    for tk in tickers:
        if tk not in df_dict:
            earnings_dict[tk] = set()
            continue
        try:
            yft = yf.Ticker(tk)
            earnings_df = yft.get_earnings_dates(limit=20)
            if earnings_df is not None and not earnings_df.empty:
                if isinstance(earnings_df.index, pd.DatetimeIndex):
                    earnings_dates = set(earnings_df.index.normalize())
                elif 'Earnings Date' in earnings_df.columns:
                    earnings_dates = set(pd.to_datetime(earnings_df['Earnings Date']).dt.normalize())
                else:
                    earnings_dates = set()
            else:
                earnings_dates = set()
        except Exception as exc:
            logger.warning(f"Could not fetch earnings for {tk}: {exc}")
            earnings_dates = set()
        earnings_dict[tk] = earnings_dates

    # 6) Build a union of all timestamps across tickers (only those between start and end)
    all_times = set()
    for tk, df_tk in df_dict.items():
        if not df_tk.empty:
            for dt in df_tk.index:
                if start <= dt <= end:
                    all_times.add(dt)
    all_times = sorted(all_times)

    # Helper: Get normalized date (maintaining UTC)
    def get_date(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.normalize()

    # 7) Prepare unique trading dates and PDT mapping
    all_dates = sorted(set(get_date(ts) for ts in all_times))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    day_trade_dates: List[pd.Timestamp] = []  # For PDT counting

    # 8) Initialize positions: one trade per ticker
    positions = {tk: None for tk in tickers}
    trades_list: List[Dict[str, Any]] = []
    balance_tracker = account_balance

    # 9) Iterate over each bar timestamp
    for current_ts in all_times:
        current_date = get_date(current_ts)
        date_idx = date_to_idx[current_date]

        # (a) Update open positions
        for tk in list(positions.keys()):
            if positions[tk] is None:
                continue
            if tk not in df_dict:
                continue
            df_tk = df_dict[tk]
            if current_ts not in df_tk.index:
                continue

            pos = positions[tk]
            bar_open = float(df_tk.loc[current_ts, "Open"])
            bar_high = float(df_tk.loc[current_ts, "High"])
            bar_low = float(df_tk.loc[current_ts, "Low"])
            bar_close = float(df_tk.loc[current_ts, "Close"])

            # Check if position held for more than 72 hours (3 days)
            hours_held = (current_ts - pos["entry_ts"]) / pd.Timedelta(hours=1)
            if hours_held > 72:
                # Force stop loss to break-even if not achieved already
                new_stop = pos["entry_price"]
                if new_stop > pos["stop_price"]:
                    pos["stop_price"] = new_stop
                # Force exit immediately at bar close
                exit_price = bar_close
                exit_ts = current_ts
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                balance_tracker += pnl
                wl_flag = 1 if pnl > 0 else 0
                ret_val = pnl / (pos["entry_price"] * pos["shares"]) if pos["shares"] > 0 else 0.0

                line = (f"{tk},{pos['entry_ts']},{exit_ts},"
                        f"{pos['entry_price']:.2f},{exit_price:.2f},{pos['shares']},"
                        f"{pnl:.2f},{wl_flag},{ret_val:.5f},"
                        f"{pos['indicators']['rsi']:.3f},{pos['indicators']['rsi_std']:.3f},"
                        f"{pos['indicators']['macd']:.3f},{pos['indicators']['atr_filter']:.3f},"
                        f"{pos['indicators']['sector']:.3f},{pos['indicators']['rvol']:.3f},"
                        f"{pos['indicators']['adx_di']:.3f},{pos['indicators']['crossover']:.3f},"
                        f"{pos['held_overnights']}\n")
                with open(TRADE_LOG_FILE, "a") as fh:
                    fh.write(line)
                trades_list.append({
                    "Ticker": tk,
                    "EntryDate": pos["entry_ts"],
                    "ExitDate": exit_ts,
                    "EntryPrice": pos["entry_price"],
                    "ExitPrice": exit_price,
                    "Shares": pos["shares"],
                    "PnL": pnl,
                    "WinLoss": wl_flag,
                    "ReturnPerc": ret_val,
                    "RSI": pos["indicators"]["rsi"],
                    "RSI_STD": pos["indicators"]["rsi_std"],
                    "MACD": pos["indicators"]["macd"],
                    "ATR_Filter": pos["indicators"]["atr_filter"],
                    "Sector": pos["indicators"]["sector"],
                    "RelativeVolume": pos["indicators"]["rvol"],
                    "ADX_DI": pos["indicators"]["adx_di"],
                    "Crossover": pos["indicators"]["crossover"],
                    "HeldOvernights": pos["held_overnights"]
                })
                positions[tk] = None
                continue

            # Normal stop loss / profit checks:
            trade_open = True
            if bar_low <= pos["stop_price"]:
                exit_price = pos["stop_price"]
                trade_open = False
            elif bar_high >= pos["profit_price"]:
                exit_price = pos["profit_price"]
                trade_open = False

            if not trade_open:
                exit_ts = current_ts
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                balance_tracker += pnl
                wl_flag = 1 if pnl > 0 else 0
                ret_val = pnl / (pos["entry_price"] * pos["shares"]) if pos["shares"] > 0 else 0.0

                line = (f"{tk},{pos['entry_ts']},{exit_ts},"
                        f"{pos['entry_price']:.2f},{exit_price:.2f},{pos['shares']},"
                        f"{pnl:.2f},{wl_flag},{ret_val:.5f},"
                        f"{pos['indicators']['rsi']:.3f},{pos['indicators']['rsi_std']:.3f},"
                        f"{pos['indicators']['macd']:.3f},{pos['indicators']['atr_filter']:.3f},"
                        f"{pos['indicators']['sector']:.3f},{pos['indicators']['rvol']:.3f},"
                        f"{pos['indicators']['adx_di']:.3f},{pos['indicators']['crossover']:.3f},"
                        f"{pos['held_overnights']}\n")
                with open(TRADE_LOG_FILE, "a") as fh:
                    fh.write(line)
                trades_list.append({
                    "Ticker": tk,
                    "EntryDate": pos["entry_ts"],
                    "ExitDate": exit_ts,
                    "EntryPrice": pos["entry_price"],
                    "ExitPrice": exit_price,
                    "Shares": pos["shares"],
                    "PnL": pnl,
                    "WinLoss": wl_flag,
                    "ReturnPerc": ret_val,
                    "RSI": pos["indicators"]["rsi"],
                    "RSI_STD": pos["indicators"]["rsi_std"],
                    "MACD": pos["indicators"]["macd"],
                    "ATR_Filter": pos["indicators"]["atr_filter"],
                    "Sector": pos["indicators"]["sector"],
                    "RelativeVolume": pos["indicators"]["rvol"],
                    "ADX_DI": pos["indicators"]["adx_di"],
                    "Crossover": pos["indicators"]["crossover"],
                    "HeldOvernights": pos["held_overnights"]
                })
                if get_date(pos["entry_ts"]) == get_date(exit_ts):
                    day_trade_dates.append(get_date(exit_ts))
                positions[tk] = None

        # Determine PDT capacity using last 5 trading days
        current_day_idx = date_to_idx[get_date(current_ts)]
        if PDT:
            start_idx = max(0, current_day_idx - 4)
            five_day_window = all_dates[start_idx: current_day_idx + 1]
            recent_day_trades = sum(1 for d in day_trade_dates if d in five_day_window)
            capacity = max(0, 3 - recent_day_trades)
        else:
            capacity = 3
        if capacity <= 0:
            continue

        # For tickers without an open position, compute final indicator scores using data up to current_ts
        scores_for_day = []
        for tk in tickers:
            if positions[tk] is not None:
                continue
            if tk not in df_dict:
                continue
            df_tk = df_dict[tk]
            if current_ts not in df_tk.index:
                continue
            df_tk_up_to_now = df_tk.loc[:current_ts]
            # Limit scoring calculations to the last 7 days
            df_tk_up_to_now = df_tk_up_to_now.last('7D')
            if len(df_tk_up_to_now) < 30:
                continue
            score_val = final_score_indicators(
                        df_tk_up_to_now,
                        sector_data,
                        current_ts,
                        compare_index_df=compare_index_data,
                        volatility_threshold=volatility_threshold,
                        weights_dict=weights_dict
            )
            scores_for_day.append((tk, score_val))
        scores_for_day.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Capacity for {current_date}: {capacity}")

        # Only consider scores above threshold; limit to capacity
        to_open = [(tk, sc) for tk, sc in scores_for_day if sc > buy_score_threshold]
        to_open = sorted(to_open, key=lambda x: x[1], reverse=True)[:capacity]

        # Open trades for the selected tickers
        for tk, sc in to_open:
            df_tk = df_dict[tk]
            day_open = float(df_tk.loc[current_ts, "Open"])
            position_value = account_balance * allocation_pct
            shares = int(position_value // day_open)
            if shares <= 0:
                continue
            # Compute dynamic ATR parameters using data up to current_ts
            ds_mult, dp_mult = compute_dynamic_atr_multiplier(
                                df_tk.loc[:current_ts],
                                stop_loss_multiplier,
                                profit_target_multiplier
            )
            atr_val = compute_atr(df_tk.loc[:current_ts][["High", "Low", "Close"]], 14).iloc[-1]
            stop_price = day_open - (atr_val * ds_mult)
            profit_price = day_open + (atr_val * dp_mult)

            # Compute indicators for logging from data up to current_ts
            rsi_series_ = compute_rsi(df_tk.loc[:current_ts, "Close"], 14)
            sector_val = compute_sector_factor(sector_data, current_ts, 5, compare_index_df=compare_index_data)
            rsi_std_val = compute_rsi_std_score(rsi_series_, 14, 60, sector_val)
            macd_static = compute_macd_score(df_tk.loc[:current_ts, "Close"])
            atr_series2 = compute_atr(df_tk.loc[:current_ts][["High", "Low", "Close"]], 14)
            atr_fval = compute_atr_filter_score(df_tk.loc[:current_ts], atr_series2)
            rsi_final = compute_rsi_score(rsi_series_, sector_factor=sector_val)
            sector_mapped = (sector_val + 2.0) / 4.0
            rvol_val = compute_relative_volume_score(df_tk.loc[:current_ts])
            adx_di_static = compute_adx_di_score(df_tk.loc[:current_ts])
            sma_short = df_tk.loc[:current_ts, "Close"].rolling(10).mean()
            sma_long = df_tk.loc[:current_ts, "Close"].rolling(30).mean()
            diff_sma_ = sma_short - sma_long if len(sma_short) == len(sma_long) else None
            crossover_static = compute_crossover_score(df_tk.loc[:current_ts], 10, 30, diff_sma_)

            # Open trade by saving details in positions
            positions[tk] = {
                "entry_ts": current_ts,
                "entry_date": get_date(current_ts),
                "entry_price": day_open,
                "shares": shares,
                "stop_price": stop_price,
                "profit_price": profit_price,
                "entry_score": sc,
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
    # End of bar iteration

    # 10) After processing all bars, force close any remaining open positions.
    for tk, pos in positions.items():
        if pos is not None:
            last_bar = df_dict[tk].iloc[-1]
            exit_price = float(last_bar["Close"])
            exit_ts = last_bar.name
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            balance_tracker += pnl
            wl_flag = 1 if pnl > 0 else 0
            ret_val = pnl / (pos["entry_price"] * pos["shares"]) if pos["shares"] > 0 else 0
            line = (f"{tk},{pos['entry_ts']},{exit_ts},"
                    f"{pos['entry_price']:.2f},{exit_price:.2f},{pos['shares']},"
                    f"{pnl:.2f},{wl_flag},{ret_val:.5f},"
                    f"{pos['indicators']['rsi']:.3f},{pos['indicators']['rsi_std']:.3f},"
                    f"{pos['indicators']['macd']:.3f},{pos['indicators']['atr_filter']:.3f},"
                    f"{pos['indicators']['sector']:.3f},{pos['indicators']['rvol']:.3f},"
                    f"{pos['indicators']['adx_di']:.3f},{pos['indicators']['crossover']:.3f},"
                    f"{pos['held_overnights']}\n")
            with open(TRADE_LOG_FILE, "a") as fh:
                fh.write(line)
            trades_list.append({
                "Ticker": tk,
                "EntryDate": pos["entry_ts"],
                "ExitDate": exit_ts,
                "EntryPrice": pos["entry_price"],
                "ExitPrice": exit_price,
                "Shares": pos["shares"],
                "PnL": pnl,
                "WinLoss": wl_flag,
                "ReturnPerc": ret_val,
                "RSI": pos["indicators"]["rsi"],
                "RSI_STD": pos["indicators"]["rsi_std"],
                "MACD": pos["indicators"]["macd"],
                "ATR_Filter": pos["indicators"]["atr_filter"],
                "Sector": pos["indicators"]["sector"],
                "RelativeVolume": pos["indicators"]["rvol"],
                "ADX_DI": pos["indicators"]["adx_di"],
                "Crossover": pos["indicators"]["crossover"],
                "HeldOvernights": pos["held_overnights"]
            })
            positions[tk] = None

    final_pnl = balance_tracker - initial_balance
    logger.info(f"Intraday-1h backtest completed. Start={initial_balance:.2f}, End={balance_tracker:.2f}, PnL={final_pnl:.2f}")
    return trades_list

###############################################################################
#                     PICKING EXACTLY 15 VALID TICKERS                        #
###############################################################################
def pick_valid_ticker(stock_lib: pd.DataFrame, start_date: datetime, end_date: datetime) -> Optional[str]:
    """
    Picks one ticker at random, checks if it has >=30 intraday 30-minute bars in [start_date, end_date].
    """
    tck = random.choice(stock_lib['Ticker'].tolist())
    try:
        df = yf.download(
            tck,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="30m",
            progress=False,
            auto_adjust=True
        ).dropna()
        df = remove_timezone(df)
        if not df.empty:
            return tck
        else:
            return None
    except:
        return None


def choose_valid_15_tickers(stock_lib: pd.DataFrame, start_date: datetime, end_date: datetime,
                            max_attempts=300) -> List[str]:
    """
    Picks exactly 15 tickers that have enough 5-minute data in the range.
    """
    valid_list = []
    attempts = 0
    while (len(valid_list) < 20) and (attempts < max_attempts):
        tck = pick_valid_ticker(stock_lib, start_date, end_date)
        attempts += 1
        if tck and (tck not in valid_list):
            valid_list.append(tck)
    if len(valid_list) < 20:
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

    # --- Gradient clipping added (clipnorm=1.0)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

    model = Model(inputs=inputs, outputs=[pnl_output, win_loss_output])
    model.compile(
        optimizer=optimizer,
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

    possible_dates = list(pd.date_range('2024-11-24', '2024-12-20', freq='D'))
    if not possible_dates:
        logger.error("No valid dates in range!")
        sys.exit(1)

    def summarize_trades(trades: List[Dict[str, Any]], initial_balance=5000.0):
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
        train_start, train_end = start_date, start_date + pd.DateOffset(days=30)
        cutoff = pd.Timestamp("2025-01-21")
        train_end = min(train_end, cutoff)

        if train_start >= train_end:
            logger.warning("Train window invalid, skipping iteration.")
            continue

        tickers_chosen = choose_valid_15_tickers(stock_lib, train_start, train_end, max_attempts=200)
        if len(tickers_chosen) < 10:
            logger.warning(f"Only {len(tickers_chosen)} valid tickers found after replacements. Proceeding.")

        # Run backtests on training data.
        train_trades = backtest_intraday_strategy(
            tickers_chosen, train_start, train_end, sector_etf="QQQ", compare_index_etf="SPY",
            volatility_threshold=1.0, buy_score_threshold=0.7, account_balance=5000.0,
            allocation_pct=0.07, stop_loss_multiplier=1.5, profit_target_multiplier=3.0, weights_dict=base_weights
        )
        if not train_trades:
            logger.info("No trades or no valid data in training period => skip iteration.")
            with open(META_LOG_FILE, "a") as f:
                f.write(f"Iteration {itx}, TRAIN => No trades\n")
            continue

        total_pnl_train, wr_train, sr_train, dd_train = summarize_trades(train_trades, 5000.0)

        log_msg_train = (f"Iteration {itx}, ({train_start} to {train_end}) TRAIN => PnL={total_pnl_train:.2f}, "
                         f"WinRate={wr_train*100:.2f}%, Sharpe={sr_train:.3f}, Drawdown={dd_train*100:.2f}%\n")

        logger.info(log_msg_train.strip())

        with open(META_LOG_FILE, "a") as f:
            f.write(log_msg_train)

        # Adjust the learning rate.
        decayed_lr = exponential_decay_lr(initial_lr, itx, decay_rate, min_lr)
        combined_sharpe = (sr_train)

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
        # ===== New: Build a training dataset from the training trades =====
        # Always initialize the training lists so they are always defined.
        X_train = []
        y_train_pnl = []
        y_train_win = []
        
        # Process trades if any are available.
        if train_trades:
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
        
        # Always check if X_train is defined (even if empty)
        # Ensure X_train is not empty
        if X_train:
            X_train = np.array(X_train)
            y_train_pnl = np.array(y_train_pnl)
            y_train_win = np.array(y_train_win)

            # Check for zero-variance columns
            std_dev = X_train.std(axis=0)
            nonzero_cols = np.where(std_dev > 1e-8)[0]  # Keep columns with variance > epsilon
            if len(nonzero_cols) == 0:
                logger.warning("All columns in X_train have zero variance; skipping training iteration.")
                continue  # Skip this iteration if no usable data

            # Filter out zero-variance columns
            X_train = X_train[:, nonzero_cols]

            # Rebuild the model if input_dim changes
            current_input_dim = X_train.shape[1]
            if current_input_dim != input_dim:
                input_dim = current_input_dim
                model = build_neural_model(input_dim, learning_rate=current_lr)

            # Scale the training features
            X_train_scaled = scaler.fit_transform(X_train)

            # Train the model
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
'''
def main():
    
    logger.info("=== Starting Revised Neural Score Optimization ===")
    final_w = iterative_optimization(
        stock_library_csv="stock_library.csv",
        iterations=200,  # Adjust iteration count as desired
        base_weights={"rsi": 0.1346593201160431,
                      "rsi_std": 0.12247657030820847,
                      "macd": 0.14777715504169464,
                      "atr_filter": 0.12393946200609207,
                      "sector": 0.11621066927909851,
                      "rvol": 0.1337585151195526,
                      "adx_di": 0.11782931536436081,
                      "crossover": 0.1033490002155304}
    )
    logger.info(f"Final Intraday Weights after all iterations: {final_w}")
    with open("intraday_optimized_weights.json", "w") as f:
        json.dump(final_w, f)
    logger.info("Saved final weights to intraday_optimized_weights.json.")

if __name__ == "__main__":
    main()
'''
def main():
    logger.info("=== Starting Backtest ===")

    # Set the backtest parameters
    start_date = datetime(2024, 12, 1)
    end_date = datetime(2025, 1, 20)

    # Load all tickers from stock_library.csv
    try:
        stock_lib = pd.read_csv("stock_library.csv")
        if "Ticker" not in stock_lib.columns:
            raise ValueError("CSV must have a 'Ticker' column.")
        tickers = stock_lib["Ticker"].dropna().unique().tolist()
        if not tickers:
            raise ValueError("No tickers found in stock_library.csv.")
    except Exception as e:
        logger.error(f"Error loading tickers from stock_library.csv: {e}")
        sys.exit(1)

    # Initial weights for the indicators
    base_weights = {"rsi": 0.12442881613969803, "rsi_std": 0.12087585777044296, "macd": 0.13195861876010895, "atr_filter": 0.11658483743667603, "sector": 0.11400578916072845, "rvol": 0.13725993037223816, "adx_di": 0.12422795593738556, "crossover": 0.13065817952156067}

    # Perform backtest
    trades = backtest_intraday_strategy(
        tickers=tickers,
        start=start_date,
        end=end_date,
        sector_etf="QQQ",
        compare_index_etf="SPY",
        volatility_threshold=1.0,
        buy_score_threshold=0.7,
        account_balance=5000.0,
        allocation_pct=0.07,
        stop_loss_multiplier=1.5,
        profit_target_multiplier=3.0,
        weights_dict=base_weights
    )

if __name__ == "__main__":
    main()
