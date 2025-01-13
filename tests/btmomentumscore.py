#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Example Backtrader script for a Momentum-Scoring Strategy (Intraday Option)
# ---------------------------------------------------------------------------
# This version:
#  1. Downloads data via yfinance and normalizes/reshapes it for Backtrader.
#  2. Optionally uses intraday data (15m or 1h) if desired.
#  3. Defines custom indicators for:
#       - RSI (with dynamic thresholding)
#       - ATR (14-day)
#       - Multi-MA (20/50/200 alignment)
#       - MACD (via built-in MACDHisto) with a bullish-check
#       - A crossover indicator (SMA10 vs. SMA30)
#       - Sector Performance & Daily Sector Change (with optional defaults)
#       - FinalScore that aggregates these partial scores.
#  4. Strategy logic: In next(), logs data and then checks:
#       - Final score threshold,
#       - MACD bullish condition, and a positive SMA crossover
#       before buying.
#       - Positions are managed with an ATR-based trailing stop.
#  5. Logs all trade events.
#
###############################################################################

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

# External libraries
import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)

###############################################################################
#                         LOGGING SETUP                                       #
###############################################################################
LOG_FILE = "log_scoretest_bt.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='w')
    ]
)
logger = logging.getLogger(__name__)

###############################################################################
#                  CUSTOM INDICATORS FOR SCORING LOGIC                        #
###############################################################################
class RSIEnhanced(bt.Indicator):
    """
    A custom RSI with dynamic thresholding for scoring.
    Lines:
      - rsi: the standard RSI
      - rsi_score: a scored value (e.g. 2.0 = strongly bullish, -2.0 = strongly overbought)
    """
    lines = ('rsi', 'rsi_score',)
    params = (('period', 14),)
    
    def __init__(self):
        self.l.rsi = bt.indicators.RSI_SMA(self.data, period=self.p.period)
    
    def next(self):
        rsi_val = self.l.rsi[0]
        if rsi_val < 25:
            self.l.rsi_score[0] = 2.0
        elif 25 <= rsi_val < 40:
            self.l.rsi_score[0] = 1.5
        elif 40 <= rsi_val < 60:
            self.l.rsi_score[0] = 1.0
        elif 60 <= rsi_val < 75:
            self.l.rsi_score[0] = 0.0
        else:
            self.l.rsi_score[0] = -2.0

class MultiMovingAverageScore(bt.Indicator):
    """
    Checks alignment of short (20), medium (50), and long (200) SMAs.
    Line:
      - ma_score: 2.0 if short > med > long, 1.0 if short > med, else 0.
    """
    lines = ('ma_score',)
    params = dict(period_short=20, period_medium=50, period_long=200)
    
    def __init__(self):
        self.sma_short = bt.indicators.SMA(self.data, period=self.p.period_short)
        self.sma_med   = bt.indicators.SMA(self.data, period=self.p.period_medium)
        self.sma_long  = bt.indicators.SMA(self.data, period=self.p.period_long)
    
    def next(self):
        short = self.sma_short[0]
        med = self.sma_med[0]
        long_ = self.sma_long[0]
        if short > med > long_:
            self.l.ma_score[0] = 2.0
        elif short > med:
            self.l.ma_score[0] = 1.0
        else:
            self.l.ma_score[0] = 0.0

class ATRVolatilityScore(bt.Indicator):
    """
    ATR-based scoring.
    Lines:
      - atr: the Average True Range.
      - atr_score: scoring based on ATR (e.g., < 0.5 -> -2, 0.8-1.5 -> +2, else 0.7).
    """
    lines = ('atr', 'atr_score',)
    params = (('period', 14),)
    
    def __init__(self):
        self.atr_ind = bt.indicators.AverageTrueRange(self.data, period=self.p.period)
    
    def next(self):
        atr_val = self.atr_ind[0]
        self.l.atr[0] = atr_val
        if atr_val < 0.5:
            self.l.atr_score[0] = -2.0
        elif 0.8 < atr_val < 1.5:
            self.l.atr_score[0] = 2.0
        elif atr_val > 3.0:
            self.l.atr_score[0] = 2.0
        else:
            self.l.atr_score[0] = 0.7

class SectorDayChange(bt.Indicator):
    """
    Returns today's sector daily change.
    Line:
      - sector_day_change: +1.5 if today's close ≥ yesterday's, else 0.
    """
    lines = ('sector_day_change',)
    
    def __init__(self, sector_data=None):
        self.sector_data = sector_data if sector_data is not None else self.data
    
    def next(self):
        if len(self.sector_data) < 2:
            self.l.sector_day_change[0] = 0.0
            return
        close0 = self.sector_data.close[0]
        close1 = self.sector_data.close[-1]
        change = (close0 - close1) / close1 if close1 != 0 else 0
        self.l.sector_day_change[0] = 1.5 if change >= 0 else 0.0

class SectorPerformanceFactor(bt.Indicator):
    """
    Measures long-term sector performance.
    Line:
      - sector_score: Weighted between -2.0 and +2.0 (based on a ±5% change).
    """
    lines = ('sector_score',)
    
    def __init__(self, sector_data=None):
        self.sector_data = sector_data if sector_data is not None else self.data
        self.initial = None
    
    def next(self):
        if self.initial is None and not np.isnan(self.sector_data.close[0]):
            self.initial = self.sector_data.close[0]
        if self.initial is None or self.initial <= 0:
            self.l.sector_score[0] = 0.0
            return
        curr_close = self.sector_data.close[0]
        perf = (curr_close - self.initial) / self.initial
        if perf >= 0.05:
            self.l.sector_score[0] = 2.0
        elif perf <= -0.05:
            self.l.sector_score[0] = -2.0
        else:
            self.l.sector_score[0] = (perf / 0.05) * 2.0

class MACDSlopeOrHist(bt.Indicator):
    """
    Checks if the MACD histogram is bullish.
    Line:
      - macd_ok: 1.0 if MACD histogram > 0, else 0.
    """
    lines = ('macd_ok',)
    params = dict(fast=12, slow=26, signal=9)
    
    def __init__(self):
        self.macd = bt.indicators.MACDHisto(
            self.data,
            period_me1=self.p.fast,
            period_me2=self.p.slow,
            period_signal=self.p.signal
        )
    
    def next(self):
        self.l.macd_ok[0] = 1.0 if self.macd.histo[0] > 0 else 0.0

class FinalScore(bt.Indicator):
    """
    Aggregates the partial indicators into a final score.
    Weighted components:
      - RSI score (w = 1.5)
      - ATR score (w = 1.0)
      - MA score (w = 1.5)
      - Sector Performance score (w = 1.5)
      - Sector Day Change (w = 1.0)
    """
    lines = ('score',)
    params = dict(
        w_rsi=1.5, w_atr=1.0, w_ma=1.5, w_sector_perf=1.5, w_sector_day=1.0
    )
    
    def __init__(self, rsi_ind, atr_ind, ma_ind, sector_perf, sector_day):
        self.rsi_ind = rsi_ind
        self.atr_ind = atr_ind
        self.ma_ind = ma_ind
        self.sector_perf = sector_perf
        self.sector_day = sector_day
    
    def next(self):
        rsi_sc      = self.rsi_ind.rsi_score[0]
        atr_sc      = self.atr_ind.atr_score[0]
        ma_sc       = self.ma_ind.ma_score[0]
        sec_perf_sc = self.sector_perf.sector_score[0]
        sec_day_sc  = self.sector_day.sector_day_change[0]
        wsum = (self.p.w_rsi + self.p.w_atr + self.p.w_ma +
                self.p.w_sector_perf + self.p.w_sector_day)
        total = (rsi_sc * self.p.w_rsi +
                 atr_sc * self.p.w_atr +
                 ma_sc * self.p.w_ma +
                 sec_perf_sc * self.p.w_sector_perf +
                 sec_day_sc * self.p.w_sector_day)
        self.l.score[0] = total / wsum if wsum else 0.0

###############################################################################
#                        STRATEGY DEFINITION                                  #
###############################################################################
class MomentumScoringStrategy(bt.Strategy):
    """
    Strategy that:
      1. Builds a custom final score from several indicators.
      2. Requires that final score ≥ threshold, MACD bullish signal,
         and a positive SMA crossover (SMA10 crossing above SMA30) before buying.
      3. Manages positions with an ATR-based trailing stop.
    """
    params = dict(
        buy_score_threshold=1.5,
        atr_trail_multiplier=1.5,
        fast_ma=12,
        slow_ma=26,
        signal_ma=9,
        debug=True  # Enable extra debug logging
    )
    
    def logdata(self, extra=""):
        dt = self.datas[0].datetime.date(0).isoformat()
        price = self.datas[0].close[0]
        score = self.final_score[0]
        logger.info(f"{dt} Price={price:.2f} Score={score:.2f} {extra}")
    
    def __init__(self):
        if len(self.datas) < 2:
            raise ValueError("Two data feeds required: primary symbol and sector ETF.")
        # Use 200 bars as the warm-up period
        self.minbars = 200
        
        # Primary data indicators
        self.rsi_enh  = RSIEnhanced(self.datas[0], period=14)
        self.atr_sc   = ATRVolatilityScore(self.datas[0], period=14)
        self.ma_sc    = MultiMovingAverageScore(self.datas[0])
        self.macd_check = MACDSlopeOrHist(
            self.datas[0],
            fast=self.p.fast_ma,
            slow=self.p.slow_ma,
            signal=self.p.signal_ma
        )
        # Crossover indicator using SMA(10) and SMA(30)
        self.crossover = bt.indicators.CrossOver(
            bt.indicators.SMA(self.datas[0], period=10),
            bt.indicators.SMA(self.datas[0], period=30)
        )
        # Sector-based indicators from second data feed
        self.sector_perf = SectorPerformanceFactor(self.datas[1])
        self.sector_day  = SectorDayChange(self.datas[1])
        # Final score aggregator
        self.final_score = FinalScore(
            rsi_ind=self.rsi_enh,
            atr_ind=self.atr_sc,
            ma_ind=self.ma_sc,
            sector_perf=self.sector_perf,
            sector_day=self.sector_day
        )
        # ATR for trailing stop management
        self.atr_val = bt.indicators.ATR(self.datas[0], period=14)
        self.order = None
        self.stop_price = None
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"BUY EXECUTED: Price={order.executed.price:.2f}, Cost={order.executed.value:.2f}")
                self.stop_price = order.executed.price - self.p.atr_trail_multiplier * self.atr_val[0]
            else:
                logger.info(f"SELL EXECUTED: Price={order.executed.price:.2f}, Cost={order.executed.value:.2f}")
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning('Order Canceled/Margin/Rejected')
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            logger.info(f"OPERATION PROFIT, GROSS {trade.pnl:.2f}")
    
    def next(self):
        # Only process if warm-up is met
        if len(self.datas[0]) < self.minbars:
            return
        
        price = self.datas[0].close[0]
        score = self.final_score[0]
        
        # Log current values (even if NaN)
        extra_debug = ""
        if self.p.debug:
            extra_debug = (
                f"RSI_Score={self.rsi_enh.rsi_score[0]:.2f}, "
                f"MA_Score={self.ma_sc.ma_score[0]:.2f}, "
                f"ATR_Score={self.atr_sc.atr_score[0]:.2f}, "
                f"Sector_Score={self.sector_perf.sector_score[0]:.2f}, "
                f"Sector_Day={self.sector_day.sector_day_change[0]:.2f}, "
                f"MACD_OK={self.macd_check.macd_ok[0]:.2f}, "
                f"Crossover={self.crossover[0]:.2f}"
            )
        self.logdata(extra_debug)
        
        # Only trade if price and score are valid
        if np.isnan(price) or np.isnan(score):
            return
        
        if self.order:
            return
        
        pos = self.getposition(self.datas[0])
        # Update trailing stop if in position
        if pos and pos.size > 0:
            cur_stop = self.datas[0].close[0] - self.p.atr_trail_multiplier * self.atr_val[0]
            if cur_stop > self.stop_price:
                self.stop_price = cur_stop
            if self.datas[0].close[0] < self.stop_price:
                logger.info(f"Trigger SELL due to trailing stop: close={self.datas[0].close[0]:.2f}, stop={self.stop_price:.2f}")
                self.order = self.sell()
                return
        
        # Check entry conditions: final score, MACD bullish, and positive SMA crossover.
        final_val = self.final_score[0]
        macd_ok = (self.macd_check.macd_ok[0] > 0)
        crossover_ok = (self.crossover[0] > 0)
        if not pos and final_val >= self.p.buy_score_threshold and macd_ok and crossover_ok:
            logger.info(f"Placing BUY order. Score={final_val:.2f}, MACD_OK={macd_ok}, Crossover={crossover_ok}")
            self.order = self.buy()
    
    def stop(self):
        logger.info("Strategy ending Value: %.2f" % self.broker.getvalue())

###############################################################################
#                           DATA PROCESSING HELPER                            #
###############################################################################
def flatten_df_columns(df):
    # Ensure the index is a proper DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Error converting index to datetime: {e}")
    # Process column names (do not reset the index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(el) for el in col]).strip() for col in df.columns.values]
    else:
        df.columns = df.columns.str.lower()
    # Rename any column named "date" to "datetime" if present (but leave index intact)
    if "date" in df.columns:
        df.rename(columns={"date": "datetime"}, inplace=True)
    # Force numeric conversion on OHLC and volume columns
    for col in ["open", "high", "low", "close", "adj close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Use original "close" column if it contains valid data;
    # otherwise, if "adj close" is available and valid, use it.
    if "close" in df.columns and not df["close"].isna().all():
        pass
    elif "adj close" in df.columns and not df["adj close"].isna().all():
        df["close"] = df["adj close"]
    return df

###############################################################################
#                           MAIN / TEST CODE                                  #
###############################################################################
def run_backtest(tickers=None, sector_symbol="XLK", start=None, end=None,
                 cash=100000.0, intraday=False, interval_override=None):
    """
    Fetch data from yfinance, process it, and run in Backtrader.
    If intraday is True, download intraday data using the specified interval.
    Note: Yahoo Finance restricts intraday data as follows:
          • '15m' data: only available for the last 60 days.
          • '1h' data: available for the last 730 days.
    You may override the interval by providing `interval_override`.
    """
    if not tickers:
        tickers = ["KOPN"]
    main_symbol = tickers[0]
    if start is None:
        start = datetime(2020, 1, 1)
    if end is None:
        end = datetime(2023, 12, 31)
    
    # Determine the interval and timeframe
    if intraday:
        interval = interval_override if interval_override else "15m"
        timeframe = bt.TimeFrame.Minutes
        if interval == "15m":
            compression = 15
            max_days = 60
        elif interval in ["60m", "1h"]:
            compression = 60
            max_days = 730
        else:
            interval = "1d"
            timeframe = bt.TimeFrame.Days
            compression = 1
            max_days = None

        if max_days is not None:
            today = datetime.now()
            allowed_start = today - timedelta(days=max_days)
            if start < allowed_start:
                logger.warning(f"For {interval} data, Yahoo Finance only supports data for the last {max_days} days. Adjusting start date from {start.date()} to {allowed_start.date()}.")
                start = allowed_start
    else:
        interval = "1d"
        timeframe = bt.TimeFrame.Days
        compression = 1

    logger.info(f"Downloading data for main symbol: {main_symbol} ({interval} data)")
    df_main = yf.download(main_symbol, start=start, end=end, interval=interval, progress=False)
    if df_main.empty:
        logger.error("Main symbol data is empty. Exiting.")
        return
    # Keep the DatetimeIndex intact.
    df_main = flatten_df_columns(df_main)
    logger.info(f"Downloaded {len(df_main)} rows for {main_symbol}")

    logger.info(f"Downloading data for sector symbol: {sector_symbol} ({interval} data)")
    df_sector = yf.download(sector_symbol, start=start, end=end, interval=interval, progress=False)
    if df_sector.empty:
        logger.warning("Sector data is empty. Proceeding with default (zero) values.")
    else:
        df_sector = flatten_df_columns(df_sector)
    logger.info(f"Downloaded {len(df_sector)} rows for {sector_symbol}")

    cerebro = bt.Cerebro()
    cerebro.addstrategy(MomentumScoringStrategy, debug=True)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)

    data_main = bt.feeds.PandasData(
        dataname=df_main,
        name=main_symbol,
        timeframe=timeframe,
        compression=compression
    )
    data_sector = bt.feeds.PandasData(
        dataname=df_sector,
        name=sector_symbol,
        timeframe=timeframe,
        compression=compression
    )
    cerebro.adddata(data_main)    # datas[0]
    cerebro.adddata(data_sector)  # datas[1]
    
    logger.info("Starting backtest ...")
    results = cerebro.run()
    final_val = cerebro.broker.getvalue()
    logger.info(f"Final Portfolio Value: {final_val:.2f}")
    # Optionally plot:
    # cerebro.plot()
    return results

def main():
    logger.info("=== Starting Momentum Strategy with Backtrader ===")
    run_backtest(
        tickers=["TSLA"],  # Use TSLA as an example ticker
        sector_symbol="XLK",
        start=datetime(2023, 1, 15),
        end=datetime(2025, 1, 9),
        cash=50000.0,
        intraday=True,
        interval_override="1h"
    )
    logger.info("=== Strategy Completed ===")

if __name__ == "__main__":
    main()
