"""
scoretest.py

Objective:
    1) Use a (predefined or dynamic) list of small-cap US tech stocks.
    2) Score them using multiple indicators:
         - RSI (14-day)
         - ATR (14-day) for volatility
         - Multi-MA alignment (20, 50, 200-day moving averages)
         - A sector performance factor (from an ETF, e.g. XLK)
         - A breakout factor based on recent percent change (via breakout query)
    3) Combine these indicators into a final "momentum" (or "confidence") score.
    4) Backtest the strategy by simulating intraday trades – if the score exceeds a threshold,
       BUY at the same day's open and SELL intraday (exit if the day's low reaches the dynamic stop-loss
       or if the day's high reaches a dynamic profit target; otherwise exit at the day's close).
       No positions are held overnight.
         - Position sizing is determined based on a simulated account balance (e.g., 5% per trade).
    5) (Optionally) Execute live/paper trades via Alpaca.
    6) Log all orders/trades along with abbreviated trade details and pnl.
    7) Automatically exit positions near market close (in live mode).

Switch between backtesting and live/paper mode by setting the `backtest_mode` flag.
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

# Alpaca
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

###############################################################################
#                              LOGGING SETUP                                  #
###############################################################################
LOG_FILE = "log_scoretest.log"
TRADE_LOG_FILE = "scoretrade.txt"  # Changed file extension to .txt

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
# Formatter to keep it short (without timestamp).
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
    if len(prices) < period + 1:
        return pd.Series([50] * len(prices), index=prices.index)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

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
    if short_ma_val > med_ma_val > long_ma_val:
        return 2.0
    elif short_ma_val > med_ma_val:
        return 1.0
    else:
        return 0.0

def compute_sector_factor(sector_perf: float) -> float:
    if sector_perf >= 0.05:
        return 2.0
    elif sector_perf <= -0.05:
        return -2.0
    else:
        return (sector_perf / 0.05) * 2

def final_score_indicators(
    df: pd.DataFrame,
    sector_perf: float = 0.0,
    volatility_threshold: float = 1.0,
    sector_day_change: float = 0.0
) -> float:
    if len(df) < 20:
        return 0.0

    rsi_val = compute_rsi(df["Close"], period=14).iloc[-1]
    atr_val = compute_atr(df[["High", "Low", "Close"]], period=14).iloc[-1]
    ma_score = compute_multi_ma_score(df)
    sector_score = compute_sector_factor(sector_perf)
    
    # RSI scoring remains unchanged (per your request)
    if rsi_val <= 30:
        rsi_score = 2.0
    elif 40 <= rsi_val < 55:
        rsi_score = 1.5
    elif rsi_val >= 70:
        rsi_score = -2.0
    else:
        rsi_score = 1.0

    # ATR scoring remains unchanged
    if atr_val < 0.5:
        atr_score = -2.0
    elif atr_val > 3.0 and 60 <= rsi_val < 70:
        atr_score = 2.0
        rsi_score = 2.0
    elif 0.8 < atr_val < 1.5 and rsi_score == 1.5:
        atr_score = 2.0
        rsi_score = 2.0
    else:
        atr_score = 0.7

    # Compute sector day score: if the day's ETF close change is nonnegative, use 2.0; else 0.0.
    
    if sector_day_change >= 0.0:
        sector_day_score = 1.5 
    else:
        sector_day_score = 0.0

    # Weighted components: MA, RSI, ATR, long-term sector factor, and sector day change.
    weights = [1.5, 1.5, 1.0, 1.5, 1.5]
    components = [ma_score, rsi_score, atr_score, sector_score, sector_day_score]
    final_score = sum(s * w for s, w in zip(components, weights)) / sum(weights)
    return final_score

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
#  BACKTESTING FUNCTION WITH DYNAMIC STOP-LOSS, TRAILING, & DYNAMIC TARGET      #
###############################################################################
def backtest_strategy(
    tickers: List[str],
    start: datetime,
    end: datetime,
    sector_etf: str = "XLK",
    volatility_threshold: float = 1.0,
    holding_period: int = 1,  # All trades are intraday
    buy_score_threshold: float = 1.5,
    account_balance: float = 50000.0,
    allocation_pct: float = 0.05,   # 5% of account per trade
    stop_loss_multiplier: float = 1.5,  # Stop loss = entry - (ATR * multiplier)
    profit_target_multiplier: float = 3.0  # Profit target = entry + (ATR * multiplier)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Backtest the momentum strategy intraday.
      - For each trading day, using historical data up to (but not including) that day,
        compute indicators and MACD for confirmation.
      - Additionally, compute the sector ETF’s day change (today’s close versus yesterday’s).
      - Only if the MACD histogram is positive and the final score meets/exceeds the threshold,
        simulate a BUY at the current day's open.
      - Simulate the intraday exit:
            • Exit at the stop-loss if the day's low ≤ (entry - ATR*multiplier).
            • Else if the day's high reaches the profit target (entry + ATR*multiplier), exit at that level.
            • Otherwise, exit at the day's close.
      - No positions are held overnight.
      - Position sizing is fixed at 5% of the account balance.
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

    # Fetch sector data for the entire backtest period.
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

            # Compute long-term sector performance (using data up to signal_date)
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

            # Compute sector day change: compare today's ETF close with yesterday’s
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

            score_val = final_score_indicators(history, sector_perf=sector_perf, 
                                               volatility_threshold=volatility_threshold,
                                               sector_day_change=sector_day_change)

            # Confirm with MACD: require positive histogram
            macd_line, signal_line, hist_val = compute_macd(history["Close"])
            if hist_val.iloc[-1] <= 0:
                continue

            if score_val >= buy_score_threshold:
                entry_date = signal_date
                entry_price = df["Open"].loc[entry_date]
                position_value = account_balance * allocation_pct
                shares = int(position_value // entry_price)
                if shares <= 0:
                    continue

                atr_val = compute_atr(history[["High", "Low", "Close"]], period=14).iloc[-1]
                stop_price = entry_price - (atr_val * stop_loss_multiplier)
                profit_target = entry_price + (atr_val * profit_target_multiplier)

                trade_entry = {
                    "Date": entry_date,
                    "Action": "BUY",
                    "Price": float(entry_price),
                    "Shares": shares,
                    "Score": float(score_val),
                    "RSI": float(compute_rsi(history["Close"], period=14).iloc[-1]),
                    "ATR": float(atr_val),
                    "MA_Score": compute_multi_ma_score(history),
                    "Sector_Perf": sector_perf,
                    "StopLoss": float(stop_price),
                    "ProfitTarget": float(profit_target)
                }
                trades[ticker].append(trade_entry)
                trade_logger.info(
                    f"B|{ticker}|{entry_date.strftime('%Y%m%d')}|p:{entry_price:.2f}|q:{shares}|sc:{score_val:.2f}|"
                    f"r:{trade_entry['RSI']:.2f}|a:{trade_entry['ATR']:.2f}|m:{trade_entry['MA_Score']:.2f}|"
                    f"s:{sector_perf:.4f}|sl:{stop_price:.2f}|pt:{profit_target:.2f}"
                )

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
                    exit_action = "TP"
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
                trade_logger.info(f"S|{ticker}|{exit_date.strftime('%Y%m%d')}|p:{exit_price:.2f}|q:{shares}|a:{exit_action}")
                trade_logger.info(f"PNL|{ticker}|bd:{entry_date.strftime('%Y%m%d')}|sd:{exit_date.strftime('%Y%m%d')}|pnl:{pnl:.2f}")
                account_balance += pnl

                # Skip to next day to avoid multiple trades on the same day
                next_day = entry_date + timedelta(days=1)
                i = np.searchsorted(df.index, next_day)
    total_pnl = account_balance - initial_balance
    logger.info(f"Backtest completed. Starting balance: ${initial_balance:.2f}, Ending balance: ${account_balance:.2f}, Total PnL: ${total_pnl:.2f}")
    return trades

###############################################################################
#                                 MAIN                                        #
###############################################################################
def main(backtest_mode: bool = True):
    logger.info("=== Starting Momentum Strategy ===")
    api = None

    # Expanded list of small-cap tech stocks
    selected_tickers = ["BMTX", "GOAI", "KOPN", "MSAI", "QRVO", "SPSC", "BMI", "MARA", "BMTX", "ZENV", "NYAX", "BKKT", "WKEY", "KOPN"]

    logger.info("Running BACKTEST mode...")
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 1, 1)
    trades = backtest_strategy(selected_tickers, start_date, end_date)

    logger.info("=== Momentum Strategy Completed ===")

if __name__ == "__main__":
    main(backtest_mode=True)
