#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intraday_neuralscore.py

Enhanced version with:
 - Logging of raw indicators in TRADE_LOG_FILE.
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
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

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

logger = logging.getLogger("IntradayNeuralScoreLogger")

# Mute yfinance logger
logger_yf = logging.getLogger('yfinance')
logger_yf.disabled = True
logger_yf.propagate = False

TRADE_LOG_FILE = "intraday_trade_details.csv"  # CSV to which backtest trades (with indicators) are logged
META_LOG_FILE = "intraday_meta_log.txt"        # Text file for iteration-level summaries

###############################################################################
#                          TIME ZONE HELPER                                   #
###############################################################################
def remove_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force the dataframe's DateTimeIndex to be tz-naive if it's tz-aware.
    This prevents tz-related comparison issues in pandas.
    """
    if not df.empty and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


###############################################################################
#                        INDICATOR & HELPER FUNCTIONS                         #
###############################################################################
def compute_percentile_rank(series: pd.Series, value: float) -> float:
    if len(series) < 2:
        return 0.5
    return (series < value).mean()


def compute_macd(prices: pd.Series, fast_period=12, slow_period=26, signal_period=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
        bullish_cross = False
        bearish_cross = False

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


def compute_sector_factor(sector_df: pd.DataFrame, signal_date: pd.Timestamp,
                          rolling_window=5, compare_index_df: Optional[pd.DataFrame] = None) -> float:
    """
    Computes sector performance over a rolling window. Output in [-2,2] range.
    Uses a smoothing to reduce intraday noise. Ensures tz-naive comparison.
    """
    if sector_df.empty or "Close" not in sector_df.columns:
        return 0.0

    # Simple smoothing to reduce intraday noise
    sector_df_smooth = sector_df.copy()
    sector_df_smooth["Close"] = sector_df_smooth["Close"].rolling(3).mean().fillna(method="bfill")

    # Force tz-naive
    sector_df_smooth = remove_timezone(sector_df_smooth)

    # Also ensure signal_date is tz-naive
    if signal_date.tzinfo is not None:
        signal_date = signal_date.replace(tzinfo=None)

    # If signal_date not in index, pick the closest earlier date
    # (compare requires both tz-naive)
    possible_dates = sector_df_smooth.index[sector_df_smooth.index <= signal_date]
    if possible_dates.empty:
        return 0.0
    use_date = possible_dates[-1]
    pos = sector_df_smooth.index.get_loc(use_date)

    start_pos = max(pos - rolling_window + 1, 0)
    segment = sector_df_smooth["Close"].iloc[start_pos:pos + 1]

    # If segment is empty or first element == 0 (in case of Series)
    if segment.empty:
        base_perf = 0.0
    else:
        val0 = segment.iloc[0]
        # If we suspect multiple columns or array-like, check .all()
        if isinstance(val0, (float, int)):
            if val0 == 0:
                base_perf = 0.0
            else:
                base_perf = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]
        else:
            # Fallback if not purely numeric or multiple columns
            if (val0 == 0).all():
                base_perf = 0.0
            else:
                base_perf = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]

    sector_score = np.clip(base_perf * 5, -1.0, 1.0)

    # Compare with index if provided
    if (compare_index_df is not None) and ("Close" in compare_index_df.columns):
        compare_index_smooth = compare_index_df.copy()
        compare_index_smooth["Close"] = compare_index_smooth["Close"].rolling(3).mean().fillna(method="bfill")
        compare_index_smooth = remove_timezone(compare_index_smooth)

        idx_dates = compare_index_smooth.index[compare_index_smooth.index <= signal_date]
        if idx_dates.empty:
            index_perf = 0.0
        else:
            sdate2 = idx_dates[-1]
            pos_idx = compare_index_smooth.index.get_loc(sdate2)
            start_idx = max(pos_idx - rolling_window + 1, 0)
            comp_segment = compare_index_smooth["Close"].iloc[start_idx:pos_idx+1]
            val_c0 = comp_segment.iloc[0]

            if isinstance(val_c0, (float, int)):
                if val_c0 == 0:
                    index_perf = 0.0
                else:
                    index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]
            else:
                if (val_c0 == 0).all():
                    index_perf = 0.0
                else:
                    index_perf = (comp_segment.iloc[-1] - comp_segment.iloc[0]) / comp_segment.iloc[0]

        relative_perf = base_perf - index_perf
        sector_score += np.clip(relative_perf * 2, -0.5, 0.5)

    # Map from [-2..2] => [0..1]
    return float(np.clip(sector_score * 20, -2.0, 2.0))


def compute_max_drawdown(trades: List[Dict[str, Any]], initial_balance: float = 50000.0) -> float:
    """
    Compute maximum drawdown from a list of trades.
    We'll order by exit datetime and track equity after each exit.
    """
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


###############################################################################
#                              FINAL SCORE FUNC                               #
###############################################################################
def final_score_indicators(
    df: pd.DataFrame,
    sector_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    compare_index_df: Optional[pd.DataFrame] = None,
    volatility_threshold: float = 1.0,
    weights_dict: Optional[Dict[str, float]] = None,
    return_components: bool = False
) -> Any:
    """
    Combine indicators into a final score [0..1] for intraday signals.
    Optionally return raw indicator values as a dict (for logging/ML).
    """
    if weights_dict is None:
        weights_dict = {
            'rsi': 1.0, 'rsi_std': 0.5, 'macd': 1.0, 'atr_filter': 1.0,
            'sector': 1.0, 'rvol': 1.0, 'multi_ma': 1.0, 'crossover': 1.0
        }

    if len(df) < 30:
        empty_indicators = {
            "rsi": 0.5, "rsi_std": 0.5, "macd": 0.5, "atr_filter": 0.5,
            "sector": 0.5, "rvol": 0.5, "multi_ma": 0.5, "crossover": 0.5
        }
        return (0.5, empty_indicators) if return_components else 0.5

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

    # sector_score is in [-2..2], map to [0..1]
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
        if return_components:
            return 0.5, indicator_scores
        return 0.5

    raw_score = weighted_sum / total_weight

    penalty_factor = 1.0
    if atr_filter_val < 0.5:
        penalty_factor -= 0.10 * volatility_threshold

    final_score = float(np.clip(raw_score * penalty_factor, 0.0, 1.0))

    if return_components:
        return final_score, indicator_scores
    else:
        return final_score


def compute_dynamic_atr_multiplier(history_df: pd.DataFrame, base_stop_mult: float, base_profit_mult: float,
                                   period=14, lookback=60) -> Tuple[float, float]:
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


def fetch_sector_data(etf_symbol: str, start: datetime, end: datetime, interval: str = "30m") -> pd.DataFrame:
    """
    Fetch sector/index data via yfinance. If 5-minute data is empty, fall back to daily.
    Ensure tz-naive index upon return.
    """
    try:
        df = yf.download(
            etf_symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        if df.empty:
            logger.warning(f"No data for {etf_symbol} at interval={interval}. Falling back to daily interval.")
            # Try daily interval
            df = yf.download(
                etf_symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True
            )
        df = remove_timezone(df)
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
    sector_etf="XLK",
    compare_index_etf="SPY",
    volatility_threshold=1.0,
    buy_score_threshold=0.5,
    account_balance=50000.0,
    allocation_pct=0.05,
    stop_loss_multiplier=1.5,
    profit_target_multiplier=3.0,
    weights_dict: Optional[Dict[str, float]] = None,
    iteration_id: int = 0
) -> List[Dict[str, Any]]:
    """
    Backtest intraday with 5-minute bars:
      - For each trading day within [start, end]:
        - Build intraday DataFrame for each ticker, walk bar by bar.
        - If score >= buy threshold => open position at that bar's open.
        - Check if stop loss or profit target is hit intraday; else close by day's last bar.
      - Log each trade, store in trades_list.
    """
    if os.path.exists(TRADE_LOG_FILE):
        os.remove(TRADE_LOG_FILE)

    logger.info(f"[Iteration {iteration_id}] Starting Intraday Backtest (30m) from {start.date()} to {end.date()} on tickers: {tickers}")
    initial_balance = account_balance

    try:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="30m",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True
        )
        # Convert to tz-naive
        if isinstance(data.columns, pd.MultiIndex):
            # multi-index for multiple tickers
            for tck in data.columns.levels[0]:
                data[tck] = remove_timezone(data[tck])
        else:
            # single ticker
            data = remove_timezone(data)
    except Exception as e:
        logger.error(f"Error downloading intraday data: {e}")
        return []

    # Sector + compare
    sector_data = fetch_sector_data(sector_etf, start, end, interval="30m")
    if sector_data.empty or "Close" not in sector_data.columns:
        sector_data = pd.DataFrame({"Close": [1.0]}, index=[start])
        logger.warning("No valid sector data. Sector performance set to neutral=0.0")

    compare_index_data = (
        fetch_sector_data(compare_index_etf, start, end, interval="30m")
        if compare_index_etf else None
    )

    trades_list = []

    # Construct a list of unique "trading days"
    all_datetimes = []
    for tck in tickers:
        try:
            t_data = data[tck].dropna()
            if t_data.empty:
                continue
            all_datetimes.append(t_data.index)
        except:
            pass

    if not all_datetimes:
        logger.warning("No intraday data found for any ticker.")
        return []

    merged_idx = pd.concat([pd.Series(idx) for idx in all_datetimes]).drop_duplicates().sort_values()
    unique_days = sorted(list(set([x.date() for x in merged_idx])))

    account_balance_tracker = account_balance

    for ticker in tickers:
        # Ensure ticker data is in data
        if ticker not in data.columns.levels[0]:
            logger.warning(f"No intraday data for ticker {ticker}. Skipping.")
            continue

        df_tck = data[ticker].dropna()
        if df_tck.empty or len(df_tck) < 30:
            logger.warning(f"Insufficient intraday data for {ticker}. Skipping.")
            continue

        for day in unique_days:
            day_str = day.strftime("%Y-%m-%d")
            day_slice = df_tck.loc[day_str].copy() if day_str in df_tck.index else None
            if day_slice is None or day_slice.empty:
                continue

            position_open = False
            entry_price = 0.0
            shares = 0
            entry_ts = None
            stop_price = 0.0
            profit_price = 0.0
            last_indicator_dict = {}

            for i in range(len(day_slice)):
                current_ts = day_slice.index[i]
                history = df_tck.loc[:current_ts]

                # Compute final score + raw indicators
                score_val, indicator_scores = final_score_indicators(
                    history, sector_data, current_ts,
                    compare_index_df=compare_index_data,
                    volatility_threshold=volatility_threshold,
                    weights_dict=weights_dict,
                    return_components=True
                )

                if not position_open:
                    if score_val >= buy_score_threshold:
                        position_value = account_balance_tracker * allocation_pct
                        bar_open_price = day_slice["Open"].iloc[i]
                        shares = int(position_value // bar_open_price)
                        if shares <= 0:
                            continue
                        entry_price = bar_open_price
                        entry_ts = current_ts
                        ds_mult, dp_mult = compute_dynamic_atr_multiplier(
                            history, stop_loss_multiplier, profit_target_multiplier
                        )
                        atr_val = compute_atr(history[["High", "Low", "Close"]], 14).iloc[-1]
                        stop_price = entry_price - (atr_val * ds_mult)
                        profit_price = entry_price + (atr_val * dp_mult)
                        position_open = True

                        with open(TRADE_LOG_FILE, "a") as fh:
                            fh.write((
                                f"{iteration_id},{ticker},{entry_ts},(Entry),"
                                f"{entry_price:.4f},NA,{shares},"
                                f"NA,NA,NA,"
                                f"RSI_SCORE={indicator_scores['rsi']:.4f},"
                                f"RSI_STD_SCORE={indicator_scores['rsi_std']:.4f},"
                                f"MACD_SCORE={indicator_scores['macd']:.4f},"
                                f"ATR_Filter_SCORE={indicator_scores['atr_filter']:.4f},"
                                f"Sector_SCORE={indicator_scores['sector']:.4f},"
                                f"RelativeVolume_SCORE={indicator_scores['rvol']:.4f},"
                                f"Multi_MA_SCORE={indicator_scores['multi_ma']:.4f},"
                                f"Crossover_SCORE={indicator_scores['crossover']:.4f}\n"
                            ))
                else:
                    # Update last_indicator_dict
                    last_indicator_dict = indicator_scores

                    # Check stop-loss or profit
                    bar_low = day_slice["Low"].iloc[i]
                    bar_high = day_slice["High"].iloc[i]
                    if bar_low <= stop_price:
                        exit_price = stop_price
                        exit_ts = current_ts
                        pnl = (exit_price - entry_price) * shares
                        account_balance_tracker += pnl
                        win_loss = 1 if pnl > 0 else 0
                        trade_return = pnl / (entry_price * shares) if shares > 0 else 0.0

                        with open(TRADE_LOG_FILE, "a") as fh:
                            fh.write((
                                f"{iteration_id},{ticker},{exit_ts},(Stop),"
                                f"{exit_price:.4f},{entry_price:.4f},{shares},"
                                f"{pnl:.4f},{win_loss},{trade_return:.5f},"
                                f"RSI_SCORE={indicator_scores['rsi']:.4f},"
                                f"RSI_STD_SCORE={indicator_scores['rsi_std']:.4f},"
                                f"MACD_SCORE={indicator_scores['macd']:.4f},"
                                f"ATR_Filter_SCORE={indicator_scores['atr_filter']:.4f},"
                                f"Sector_SCORE={indicator_scores['sector']:.4f},"
                                f"RelativeVolume_SCORE={indicator_scores['rvol']:.4f},"
                                f"Multi_MA_SCORE={indicator_scores['multi_ma']:.4f},"
                                f"Crossover_SCORE={indicator_scores['crossover']:.4f}\n"
                            ))

                        trades_list.append({
                            "Iteration": iteration_id,
                            "Ticker": ticker,
                            "EntryDate": entry_ts,
                            "ExitDate": exit_ts,
                            "EntryPrice": entry_price,
                            "ExitPrice": exit_price,
                            "Shares": shares,
                            "PnL": pnl,
                            "WinLoss": win_loss,
                            "ReturnPerc": trade_return,
                            **{k.upper(): v for k, v in indicator_scores.items()}
                        })
                        position_open = False

                    elif bar_high >= profit_price:
                        exit_price = profit_price
                        exit_ts = current_ts
                        pnl = (exit_price - entry_price) * shares
                        account_balance_tracker += pnl
                        win_loss = 1 if pnl > 0 else 0
                        trade_return = pnl / (entry_price * shares) if shares > 0 else 0.0

                        with open(TRADE_LOG_FILE, "a") as fh:
                            fh.write((
                                f"{iteration_id},{ticker},{exit_ts},(Profit),"
                                f"{exit_price:.4f},{entry_price:.4f},{shares},"
                                f"{pnl:.4f},{win_loss},{trade_return:.5f},"
                                f"RSI_SCORE={indicator_scores['rsi']:.4f},"
                                f"RSI_STD_SCORE={indicator_scores['rsi_std']:.4f},"
                                f"MACD_SCORE={indicator_scores['macd']:.4f},"
                                f"ATR_Filter_SCORE={indicator_scores['atr_filter']:.4f},"
                                f"Sector_SCORE={indicator_scores['sector']:.4f},"
                                f"RelativeVolume_SCORE={indicator_scores['rvol']:.4f},"
                                f"Multi_MA_SCORE={indicator_scores['multi_ma']:.4f},"
                                f"Crossover_SCORE={indicator_scores['crossover']:.4f}\n"
                            ))

                        trades_list.append({
                            "Iteration": iteration_id,
                            "Ticker": ticker,
                            "EntryDate": entry_ts,
                            "ExitDate": exit_ts,
                            "EntryPrice": entry_price,
                            "ExitPrice": exit_price,
                            "Shares": shares,
                            "PnL": pnl,
                            "WinLoss": win_loss,
                            "ReturnPerc": trade_return,
                            **{k.upper(): v for k, v in indicator_scores.items()}
                        })
                        position_open = False

            # End of day => if still open, close
            if position_open:
                exit_ts = day_slice.index[-1]
                exit_price = day_slice["Close"].iloc[-1]
                pnl = (exit_price - entry_price) * shares
                account_balance_tracker += pnl
                win_loss = 1 if pnl > 0 else 0
                trade_return = pnl / (entry_price * shares) if shares > 0 else 0.0
                with open(TRADE_LOG_FILE,"a") as fh:
                    fh.write((
                        f"{iteration_id},{ticker},{exit_ts},(EODExit),"
                        f"{exit_price:.4f},{entry_price:.4f},{shares},"
                        f"{pnl:.4f},{win_loss},{trade_return:.5f},"
                        f"RSI_SCORE={last_indicator_dict.get('rsi',0.5):.4f},"
                        f"RSI_STD_SCORE={last_indicator_dict.get('rsi_std',0.5):.4f},"
                        f"MACD_SCORE={last_indicator_dict.get('macd',0.5):.4f},"
                        f"ATR_Filter_SCORE={last_indicator_dict.get('atr_filter',0.5):.4f},"
                        f"Sector_SCORE={last_indicator_dict.get('sector',0.5):.4f},"
                        f"RelativeVolume_SCORE={last_indicator_dict.get('rvol',0.5):.4f},"
                        f"Multi_MA_SCORE={last_indicator_dict.get('multi_ma',0.5):.4f},"
                        f"Crossover_SCORE={last_indicator_dict.get('crossover',0.5):.4f}\n"
                    ))
                trades_list.append({
                    "Iteration": iteration_id,
                    "Ticker": ticker,
                    "EntryDate": entry_ts,
                    "ExitDate": exit_ts,
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "Shares": shares,
                    "PnL": pnl,
                    "WinLoss": win_loss,
                    "ReturnPerc": trade_return,
                    **{k.upper(): v for k, v in last_indicator_dict.items()}
                })
                position_open = False

    final_pnl = account_balance_tracker - initial_balance
    logger.info(f"[Iteration {iteration_id}] Intraday backtest completed. "
                f"Start={initial_balance:.2f}, End={account_balance_tracker:.2f}, PnL={final_pnl:.2f}")
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
def build_neural_model(input_dim: int, learning_rate: float = 0.001) -> Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu', name="dense_shared")(inputs)
    x = Dropout(0.3)(x)  # slightly higher dropout
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

    names = ["rsi","rsi_std","macd","atr_filter","sector","rvol","multi_ma","crossover"]
    out = {}
    for i, nm in enumerate(names):
        out[nm] = float(norm_w[i]) if i < len(norm_w) else 0.0
    return out


###############################################################################
#                          ITERATIVE OPTIMIZATION                             #
###############################################################################
def iterative_intraday_optimization(
    stock_library_csv: str,
    iterations: int = 3,
    base_weights: Optional[Dict[str, float]] = None
):
    """
    1) For each iteration:
       - Randomly pick a 40-day window within the last 60 days.
       - We define a train/validation split within those 40 days (e.g. 25 days train, 15 days val).
       - Pick EXACTLY 15 tickers that have enough intraday data.
       - Backtest on train => parse intraday_trade_details.csv => train neural net (with raw indicators).
       - Backtest on val => evaluate performance => log iteration results to "intraday_meta_log.txt".
       - Optionally adjust LR if validation performance is poor (dynamic LR).
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

    # Build NN
    input_dim = len(base_weights)  # 8 indicators
    current_lr = 0.001
    model = build_neural_model(input_dim, learning_rate=current_lr)
    scaler = StandardScaler()

    now = pd.Timestamp.now().normalize()
    earliest_start = now - pd.Timedelta(days=60)

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

    if os.path.exists(META_LOG_FILE):
        os.remove(META_LOG_FILE)

    for itx in range(1, iterations+1):
        logger.info(f"\n=== Intraday Iteration {itx} ===")

        max_offset = (now - earliest_start).days - 40
        if max_offset < 1:
            logger.error("Not enough room to pick 40 days from the last 60. Aborting.")
            break

        offset = random.randint(0, max_offset)
        start_date = earliest_start + pd.Timedelta(days=offset)
        end_date = start_date + pd.Timedelta(days=40)
        if end_date > now:
            end_date = now

        train_end_date = start_date + pd.Timedelta(days=25)
        val_start_date = train_end_date + pd.Timedelta(days=1)
        val_end_date = end_date

        tickers_chosen = choose_valid_15_tickers(stock_lib, start_date, end_date, max_attempts=300)
        if len(tickers_chosen) < 15:
            logger.warning(f"Only {len(tickers_chosen)} valid tickers => proceed with fewer tickers.")

        # TRAIN
        train_trades = backtest_intraday_strategy(
            tickers_chosen, start_date, train_end_date,
            sector_etf="XLK", compare_index_etf="SPY",
            volatility_threshold=1.0, buy_score_threshold=0.5,
            account_balance=50000.0, allocation_pct=0.1,
            stop_loss_multiplier=1.5, profit_target_multiplier=3.0,
            weights_dict=base_weights,
            iteration_id=itx
        )
        if not train_trades:
            logger.info(f"[Iteration {itx}] No trades in TRAIN => skip iteration.")
            with open(META_LOG_FILE, "a") as f:
                f.write(f"Iteration {itx}, TRAIN => No trades\n")
            continue

        total_pnl_train, wr_train, sr_train, dd_train = summarize_trades(train_trades, 50000.0)

        df = pd.read_csv(TRADE_LOG_FILE, header=None)
        if df.empty:
            logger.warning("Trade log empty => skip training.")
        else:
            # Check the number of columns in the CSV
            if df.shape[1] == 18:
                # These 18 columns come from the log lines that have indicator columns separated out.
                df.columns = [
                    "Iteration", "Ticker", "TradeDate", "TradeType", "Price1", "Price2",
                    "Shares", "PnL", "WinLoss", "ReturnPerc", 
                    "RSI_SCORE", "RSI_STD_SCORE", "MACD_SCORE", "ATR_Filter_SCORE", 
                    "Sector_SCORE", "RelativeVolume_SCORE", "Multi_MA_SCORE", "Crossover_SCORE"
                ]
            elif df.shape[1] == 11:
                # Fallback if using the older format with a single "Indicators" field.
                df.columns = [
                    "Iteration", "Ticker", "TradeDate", "TradeType", "ExitPrice", "EntryPrice",
                    "Shares", "PnL", "WinLoss", "ReturnPerc", "Indicators"
                ]
                # List of indicator columns expected to be parsed from the "Indicators" field
                indicator_cols = [
                    "RSI_SCORE", "RSI_STD_SCORE", "MACD_SCORE", "ATR_Filter_SCORE",
                    "Sector_SCORE", "RelativeVolume_SCORE", "Multi_MA_SCORE", "Crossover_SCORE"
                ]
                # Initialize new columns with NaN
                for col in indicator_cols:
                    df[col] = np.nan

                # Define a helper to parse an indicator entry
                def parse_indicator(item):
                    if (not item) or ("=" not in item):
                        return (None, None)
                    key, value = item.split("=")
                    return key.strip(), float(value)

                # Process each row in the 'Indicators' column
                for i in range(len(df)):
                    row_indicators = df.loc[i, "Indicators"]
                    if pd.notna(row_indicators):
                        for item in row_indicators.split(","):
                            key, value = parse_indicator(item)
                            if key in indicator_cols:
                                df.at[i, key] = value

                # Drop the now-unneeded 'Indicators' column
                df.drop(columns=["Indicators"], inplace=True)
            else:
                logger.error(f"Unexpected number of columns in TRADE_LOG_FILE: {df.shape[1]}")
                return  # or handle the error as desired

            # Now define the list of indicator columns (these column names are consistent with the 18-column format)
            indicator_cols = [
                "RSI_SCORE", "RSI_STD_SCORE", "MACD_SCORE", "ATR_Filter_SCORE",
                "Sector_SCORE", "RelativeVolume_SCORE", "Multi_MA_SCORE", "Crossover_SCORE"
            ]

            # Drop any rows with missing values in the key columns
            df = df.dropna(subset=indicator_cols + ["PnL", "WinLoss"])
            if not df.empty:
                X_feat = df[indicator_cols].values
                y_pnl = df["PnL"].values
                y_win = df["WinLoss"].values.astype(float)

                X_scaled = scaler.fit_transform(X_feat)
                model.fit(
                    X_scaled,
                    {"pnl": y_pnl, "win_loss": y_win},
                    epochs=10,
                    batch_size=8,
                    verbose=0
                )

                new_weights = extract_indicator_weights(model)
                logger.info(f"[Iteration {itx}] Updated Weights from NN: {new_weights}")
                base_weights = new_weights

            if os.path.exists(TRADE_LOG_FILE):
                os.remove(TRADE_LOG_FILE)

        # VALIDATION
        if val_start_date >= val_end_date:
            logger.warning(f"[Iteration {itx}] Validation window invalid => skip val.")
            val_trades = []
        else:
            val_trades = backtest_intraday_strategy(
                tickers_chosen, val_start_date, val_end_date,
                sector_etf="XLK", compare_index_etf="SPY",
                volatility_threshold=1.0, buy_score_threshold=0.5,
                account_balance=50000.0, allocation_pct=0.1,
                stop_loss_multiplier=1.5, profit_target_multiplier=3.0,
                weights_dict=base_weights,
                iteration_id=itx
            )

        total_pnl_val, wr_val, sr_val, dd_val = summarize_trades(val_trades, 50000.0)

        # Dynamic learning rate adjustment (example)
        if wr_val < 0.4:
            current_lr *= 0.5
            logger.info(f"[Iteration {itx}] Validation WinRate < 40%. "
                        f"Reducing LR to {current_lr}. Re-compiling model.")
            model = build_neural_model(input_dim, learning_rate=current_lr)

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

        if os.path.exists(TRADE_LOG_FILE):
            os.remove(TRADE_LOG_FILE)

    return base_weights


###############################################################################
#                                   MAIN                                      #
###############################################################################
def main():
    logger.info("=== Starting Intraday Neural Score Optimization ===")
    final_w = iterative_intraday_optimization(
        stock_library_csv="stock_library.csv",
        iterations=3,  # Adjust iteration count as desired
        base_weights={
            "rsi": 0.15, "rsi_std": 0.08, "macd": 0.13, "atr_filter": 0.17,
            "sector": 0.18, "rvol": 0.13, "multi_ma": 0.17, "crossover": 0.15
        }
    )
    logger.info(f"Final Intraday Weights after all iterations: {final_w}")
    with open("intraday_optimized_weights.json", "w") as f:
        json.dump(final_w, f)
    logger.info("Saved final weights to intraday_optimized_weights.json.")


if __name__ == "__main__":
    main()
