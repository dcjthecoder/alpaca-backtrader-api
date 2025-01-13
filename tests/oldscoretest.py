#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scoretest.py

Objective:
    1) Use a (predefined or dynamic) list of small-cap US tech stocks.
    2) Score them using multiple indicators:
         - RSI (14-day) with a Bull/Bear nuance.
         - RSI_STD as a separate measure of RSI volatility.
         - MACD (histogram focus).
         - ATR (14-day) for volatility with a dynamic multiplier.
         - Sector performance factor (from an ETF, e.g. XLK).
         - Relative Volume (short vs. long average volume).
         - Multi-MA alignment (20, 50, 200-day).
         - A granular crossover test (SMA10 vs. SMA30).
    3) Combine these indicators into a final "momentum" (or "confidence") score.
    4) Backtest the strategy by simulating intraday trades – if the score exceeds a threshold,
       BUY at the same day's open and SELL intraday (stop-loss, profit target, or close).
       No positions are held overnight.
         - Position sizing is based on a simulated account balance (e.g., 5% per trade).
    5) (Optionally) Execute live/paper trades via Alpaca.
    6) Log all orders/trades along with abbreviated trade details and pnl.
    7) Automatically exit positions near market close (in live mode).

Switch between backtesting and live/paper mode by setting backtest_mode.
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta, time
from typing import List, Dict, Any

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
TRADE_LOG_FILE = "scoretrade.txt"  # Trade log output

from logging.handlers import RotatingFileHandler

# Main logger: summary output to console and file
logger = logging.getLogger("MomentumLogger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Trade logger: abbreviated trade log (only trade-related messages)
trade_logger = logging.getLogger("TradeLogger")
trade_logger.setLevel(logging.INFO)
trade_file_handler = RotatingFileHandler(TRADE_LOG_FILE, maxBytes=5_000_000, backupCount=5)
trade_formatter = logging.Formatter("%(message)s")
trade_file_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_file_handler)

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
    """Classic RSI calculation for a given price series."""
    if len(prices) < period + 1:
        return pd.Series([50] * len(prices), index=prices.index)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_rsi_score(rsi_series: pd.Series, bull: bool = True) -> float:
    """
    Scores RSI using typical Bull/Bear market zones:
      Bull: oversold=40-50, overbought=80-90
      Bear: oversold=20-30, overbought=55-65
    """
    if rsi_series.empty:
        return 0.0
    rsi_val = rsi_series.iloc[-1]

    if bull:
        # Bull market zones
        if rsi_val < 40:
            return 2.0
        elif rsi_val < 50:
            return 1.5
        elif rsi_val < 80:
            return 1.0
        elif rsi_val < 90:
            return 0.5
        else:
            return -2.0
    else:
        # Bear market zones
        if rsi_val < 20:
            return 2.0
        elif rsi_val < 30:
            return 1.5
        elif rsi_val < 55:
            return 1.0
        elif rsi_val < 65:
            return 0.5
        else:
            return -2.0

def compute_rsi_std_score(rsi_series: pd.Series, period: int = 14) -> float:
    """
    Returns a score based on the current standard deviation of RSI.
    If RSI volatility is too high, we might penalize the score (fear of choppiness).
    If it's lower, we might reward stability.
    """
    if len(rsi_series) < period:
        return 0.0
    rsi_std_val = rsi_series.rolling(window=period).std().iloc[-1]

    # Example thresholds (tweak as desired):
    if rsi_std_val >= 5.0:
        return 0.0
    elif rsi_std_val <= 2.0:
        return 2.0
    else:
        return 1.0

def compute_macd_score(prices: pd.Series) -> float:
    """Scores MACD based on the sign of the histogram."""
    macd_line, signal_line, hist = compute_macd(prices)
    if len(hist) == 0:
        return 0.0
    if hist.iloc[-1] > 0:
        return 2.0
    else:
        return -2.0

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

def compute_multi_ma_score(df: pd.DataFrame) -> float:
    """Checks alignment of 20, 50, and 200 SMAs for a basic trend read."""
    if len(df) < 200:
        return 0.0
    short_ma = df["Close"].rolling(20).mean().iloc[-1]
    med_ma = df["Close"].rolling(50).mean().iloc[-1]
    long_ma = df["Close"].rolling(200).mean().iloc[-1]
    try:
        short_ma_val = float(short_ma)
        med_ma_val = float(med_ma)
        long_ma_val = float(long_ma)
    except Exception as e:
        logger.error(f"Error converting MA values to float: {e}")
        return 0.0

    if np.isnan(long_ma_val):
        return 0.0

    # Simple scoring approach
    if short_ma_val > med_ma_val > long_ma_val:
        return 2.0
    elif short_ma_val > med_ma_val:
        return 1.0
    else:
        return 0.0

def compute_crossover_score(df: pd.DataFrame) -> float:
    """Granular crossover test for SMA10 vs. SMA30."""
    if len(df) < 30:
        return 0.0
    sma_short = df["Close"].rolling(10).mean().iloc[-1]
    sma_long = df["Close"].rolling(30).mean().iloc[-1]
    if sma_short <= sma_long:
         return 0.0
    diff = sma_short - sma_long
    rel_diff = diff / sma_long  # Relative difference
    # If the difference is 5% or more, cap the score to 2.0.
    score = 2.0 * min(rel_diff / 0.05, 1.0)
    return score

def compute_relative_volume_score(df: pd.DataFrame, short_window=5, long_window=20) -> float:
    """
    Compares short-term avg volume vs. long-term avg volume.
    Returns a simple scale to reflect unusual volume spikes.
    """
    if len(df) < long_window:
        return 0.0
    short_avg_vol = df["Volume"].tail(short_window).mean()
    long_avg_vol = df["Volume"].tail(long_window).mean()
    if long_avg_vol <= 0:
        return 0.0

    rvol = short_avg_vol / long_avg_vol
    # Scale example:
    # >= 3 => +2, >= 2 => +1, >= 1 => +0.5, else => 0
    if rvol >= 3.0:
        return 2.0
    elif rvol >= 2.0:
        return 1.0
    elif rvol >= 1.0:
        return 0.5
    else:
        return 0.0

def compute_sector_factor(sector_perf: float) -> float:
    """
    Continuous scaling for sector performance:
      e.g., if sector_perf is 5% (0.05), then score ~ 2.0; if -5% => -2.0.
    """
    return np.clip(sector_perf * 40, -2.0, 2.0)

###############################################################################
#                  FINAL SCORE COMBINATION FUNCTION                           #
###############################################################################
def final_score_indicators(
    df: pd.DataFrame,
    sector_perf: float = 0.0,
    sector_day_change: float = 0.0,
    volatility_threshold: float = 1.0,
) -> float:
    """
    Returns a final weighted momentum/confidence score.  
    Incorporates: RSI, RSI_STD, MACD, ATR, SectorPerf, RelativeVolume, MultiMA, & a short crossover.  
    Weights are exposed in indicator_weights for easy tuning or neural net calibration.
    """
    if len(df) < 20:
        return 0.0

    # Identify bull vs. bear market from sector performance
    bull_regime = True if compute_sector_factor(sector_perf) >= 0 else False

    # Pre-compute indicator series/values
    rsi_series = compute_rsi(df["Close"], period=14)
    rsi_score_val = compute_rsi_score(rsi_series, bull=bull_regime)
    rsi_std_score_val = compute_rsi_std_score(rsi_series, period=14)
    macd_score_val = compute_macd_score(df["Close"])
    atr_val = compute_atr(df[["High", "Low", "Close"]], period=14).iloc[-1]
    multi_ma_score_val = compute_multi_ma_score(df)
    sector_score_val = compute_sector_factor(sector_perf)
    crossover_score_val = compute_crossover_score(df)
    rvol_score_val = compute_relative_volume_score(df)

    # ATR-based scoring example
    if atr_val < 0.5:
        atr_score_val = -2.0
    elif atr_val > 3.0 and 60 <= df["Close"].iloc[-1] < 80:
        atr_score_val = 2.0
        rsi_score_val = 2.0  # Overriding RSI in certain high-ATR conditions, as an example
    elif 0.8 < atr_val < 1.5 and rsi_score_val == 1.5:
        atr_score_val = 2.0
        rsi_score_val = 2.0
    else:
        atr_score_val = 1.0

    # Sector day change score (day-over-day)
    sector_day_score_val = 1.5 if sector_day_change >= 0.0 else 0.0

    # Example: gather all indicator scores
    indicator_scores = {
        "rsi": rsi_score_val,
        "rsi_std": rsi_std_score_val,
        "macd": macd_score_val,
        "atr": atr_score_val,
        "sector_perf": sector_score_val,
        "sector_day": sector_day_score_val,
        "rvol": rvol_score_val,
        "multi_ma": multi_ma_score_val,
        "crossover": crossover_score_val,
    }

    # Weights are exposed here for easy modification or neural-net calibration
    indicator_weights = {
        "rsi": 1.0,
        "rsi_std": 0.5,
        "macd": 1.2,
        "atr": 1.0,
        "sector_perf": 1.4,
        "sector_day": 1.2,
        "rvol": 1.1,
        "multi_ma": 1.5,
        "crossover": 1.0,
    }

    # Weighted sum of all indicators
    weighted_sum = sum(indicator_scores[key] * indicator_weights[key] for key in indicator_scores.keys())
    total_weights = sum(indicator_weights.values())
    final_score = weighted_sum / total_weights

    return final_score

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
    # Scale multipliers based on ratio of current ATR to its median
    dynamic_stop_mult = base_stop_mult * (current_atr / atr_median) if atr_median > 0 else base_stop_mult
    dynamic_profit_mult = base_profit_mult * (current_atr / atr_median) if atr_median > 0 else base_profit_mult
    return dynamic_stop_mult, dynamic_profit_mult

###############################################################################
#                     SECTOR PERFORMANCE FUNCTION                             #
###############################################################################
def fetch_sector_data(etf_symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        df = yf.download(
            etf_symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False
        )
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
    volatility_threshold: float = 1.0,
    holding_period: int = 1,  # Intraday trades only
    buy_score_threshold: float = 1.5,
    account_balance: float = 50000.0,
    allocation_pct: float = 0.05,   # 5% of account per trade
    stop_loss_multiplier: float = 1.5,
    profit_target_multiplier: float = 3.0
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Backtest the momentum strategy intraday with the new multi-indicator scoring system.
    """
    logger.info(f"Starting Backtest from {start.date()} to {end.date()} on tickers: {tickers}")
    initial_balance = account_balance

    try:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True
        )
    except Exception as e:
        logger.error(f"Error downloading ticker data: {e}")
        sys.exit(1)

    # Fetch sector data
    sector_data = fetch_sector_data(sector_etf, start, end)
    if sector_data.empty or "Close" not in sector_data.columns:
        logger.warning(f"No data for ETF {sector_etf}. Sector performance set to 0.0.")
        sector_data = pd.DataFrame({"Close": [1.0]}, index=[start])

    trades = {ticker: [] for ticker in tickers}

    for ticker in tickers:
        if ticker not in data.columns.levels[0]:
            logger.warning(f"No data for ticker {ticker}.")
            continue
        df = data[ticker].dropna()
        if df.empty or len(df) < 200:
            logger.warning(f"Not enough data for {ticker}; require at least 200 days for MA calculation.")
            continue

        for i in tqdm(range(200, len(df)), desc=f"Processing {ticker}", leave=False):
            signal_date = df.index[i]
            history = df.iloc[:i]

            # Compute long-term sector performance using ETF data up to signal_date
            sector_window = sector_data.loc[:signal_date]
            if len(sector_window) < 2:
                sector_perf = 0.0
            else:
                try:
                    start_px_sector = float(sector_window["Close"].iloc[0])
                    end_px_sector = float(sector_window["Close"].iloc[-1])
                except Exception as e:
                    logger.error(f"Error converting sector prices for {ticker} on {signal_date}: {e}")
                    sector_perf = 0.0
                else:
                    sector_perf = (end_px_sector - start_px_sector) / start_px_sector

            # Compute sector day change for ETF
            if signal_date in sector_data.index:
                pos = sector_data.index.get_loc(signal_date)
                if pos > 0:
                    prev_close = sector_data["Close"].iloc[pos - 1]
                    curr_close = sector_data["Close"].iloc[pos]
                    sector_day_change = float((curr_close - prev_close) / prev_close)
                else:
                    sector_day_change = 0.0
            else:
                sector_day_change = 0.0

            # Calculate the composite final score
            score_val = final_score_indicators(
                history,
                sector_perf=sector_perf,
                sector_day_change=sector_day_change,
                volatility_threshold=volatility_threshold
            )

            # Optional check: We used MACD in the final score. If you still want a separate filter:
            # macd_line, signal_line, hist_val = compute_macd(history["Close"])
            # if hist_val.iloc[-1] <= 0:
            #     continue  # e.g. require positive MACD histogram

            if score_val >= buy_score_threshold:
                entry_date = signal_date
                entry_price = df["Open"].loc[entry_date]
                position_value = account_balance * allocation_pct
                shares = int(position_value // entry_price)
                if shares <= 0:
                    continue

                # Compute dynamic ATR multipliers based on a 60-day rolling median
                dynamic_stop_mult, dynamic_profit_mult = compute_dynamic_atr_multiplier(
                    history, stop_loss_multiplier, profit_target_multiplier, period=14, lookback=60
                )
                atr_val = compute_atr(history[["High", "Low", "Close"]], period=14).iloc[-1]
                stop_price = entry_price - (atr_val * dynamic_stop_mult)
                profit_target = entry_price + (atr_val * dynamic_profit_mult)

                # Log some details
                rsi_static = compute_rsi(history["Close"], period=14).iloc[-1]
                rsi_std_val = history["Close"].pipe(compute_rsi, 14).rolling(window=14).std().iloc[-1] if len(history) >= 14 else 0
                crossover_score_val = compute_crossover_score(history)
                ma_score_val = compute_multi_ma_score(history)
                macd_score_val = compute_macd_score(history["Close"])

                trade_entry = {
                    "Date": entry_date,
                    "Action": "BUY",
                    "Price": float(entry_price),
                    "Shares": shares,
                    "Score": float(score_val),
                    "RSI": float(rsi_static),
                    "RSI_STD": float(rsi_std_val),
                    "ATR": float(atr_val),
                    "dynATR_mult": dynamic_stop_mult,
                    "MA_Score": ma_score_val,
                    "MACD_Score": macd_score_val,
                    "Sector_Perf": sector_perf,
                    "Crossover": crossover_score_val,
                    "StopLoss": float(stop_price),
                    "ProfitTarget": float(profit_target)
                }
                trades[ticker].append(trade_entry)

                day_data = df.loc[entry_date:entry_date]
                if day_data.empty:
                    continue

                # Intraday exit conditions
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
                    exit_action = "EC"  # Exit at Close
                    exit_date = day_data.index[-1]

                trade_exit = {
                    "Date": exit_date,
                    "Action": "SELL",
                    "Price": float(exit_price),
                    "Shares": shares,
                    "Score": float(score_val)
                }
                trades[ticker].append(trade_exit)
                pnl = (exit_price - entry_price) * shares
                trade_logger.info(
                    f"TRADE|ticker:{ticker}|entry_date:{entry_date.strftime('%Y%m%d')}|exit_date:{exit_date.strftime('%Y%m%d')}|"
                    f"entry_price:{entry_price:.2f}|exit_price:{exit_price:.2f}|shares:{shares}|entry_score:{score_val:.2f}|"
                    f"rsi:{rsi_static:.2f}|rsi_std:{rsi_std_val:.2f}|atr:{atr_val:.2f}|dynATR:{dynamic_stop_mult:.2f}|"
                    f"ma_score:{ma_score_val:.2f}|macd_score:{macd_score_val:.2f}|sector_perf:{sector_perf:.4f}|"
                    f"crossover:{crossover_score_val:.2f}|stop_loss:{stop_price:.2f}|profit_target:{profit_target:.2f}|"
                    f"action:{exit_action}|pnl:{pnl:.2f}"
                )

                account_balance += pnl

                # Skip to next day to avoid multiple trades on the same day
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
    logger.info("=== Starting Momentum Strategy ===")
    api = None

    # Predefined list of small-cap tech stocks
    selected_tickers = ["BMTX", "GOAI", "KOPN", "MSAI", "QRVO", "SPSC", "BMI", "MARA",
                        "ZENV", "BKKT", "WKEY", "MAPS", "MRAM", "ALAR", "AEXAF"]

    if backtest_mode:
        logger.info("Running BACKTEST mode...")
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 1, 1)
        trades = backtest_strategy(selected_tickers, start_date, end_date)
        logger.info("=== Momentum Strategy Backtest Completed ===")
    else:
        logger.info("LIVE/PAPER mode not fully implemented in this example.")
        # Implement live/paper trading logic via Alpaca or another broker API here.

if __name__ == "__main__":
    main(backtest_mode=True)