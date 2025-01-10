"""
momentum.py

Objective:
    1) Dynamically fetch a watchlist that combines:
       - US small-cap tech stocks
       - Top daily gainers in the tech sector
    2) Score them using multi-MA, RSI, Volume, ATR-based volatility
    3) Adjust buy/sell thresholds based on tech sector performance (e.g., from XLK).
    4) Run a Backtrader-based momentum strategy on Alpaca (paper or live).
    5) Ensure robust logging and data handling.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List

import yfinance as yf
import pandas as pd
import numpy as np
import backtrader as bt
import alpaca_backtrader_api
from dotenv import load_dotenv

# yahoo_fin imports
from yahoo_fin import stock_info as si

###############################################################################
#                              ENV & LOGGING                                  #
###############################################################################
LOG_FILE = "log_momentum.log"

# Configure RotatingFileHandler to prevent log files from growing indefinitely
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10**6, backupCount=5)

# Create formatter and add it to handlers
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load environment variables from env.txt
load_dotenv("env.txt")
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    logger.error("Alpaca API keys are missing in env.txt. Exiting.")
    sys.exit()

# Set flags for demonstration
IS_LIVE = False       # True -> real trading, False -> paper trading
IS_BACKTEST = False   # True -> historical backtest, False -> live/paper

###############################################################################
#                         WATCHLIST FETCHING LOGIC                             #
###############################################################################
def fetch_watchlist(
    smallcap_limit: int = 20,
    gainers_limit: int = 10,
    market_cap_threshold: float = 2e9  # 2 billion USD
) -> List[str]:
    """
    Fetch a watchlist combining US small-cap tech stocks and top daily gainers in tech.
    Utilizes yfinance to filter based on sector and market capitalization.
    """
    watchlist = set()

    # Define problematic suffixes to exclude
    problematic_suffixes = ['W', 'PR', 'LP', 'B', 'C', 'D', 'F', 'G', 'H', 
                            'J', 'K', 'N', 'O', 'Q', 'R', 'S', 'T', 'U', 
                            'V', 'Y', 'Z']

    # 1. Fetch Aggressive Small-Cap Tech Stocks
    try:
        # Use yfinance's Screener or predefined list
        # Since yfinance does not have a built-in screener, we'll use a predefined list of small-cap tech tickers
        smallcap_tech_tickers = [
            "SNX", "INTA", "FCX", "AFRM", "BTCM", "CCL", "VALE", "UAL",
            "FICO", "ITUB", "GRMN", "GILD", "RGEN", "NVDA", "TSLA",
            "ROKU", "AMZN", "MDB", "DOCU", "SQ", "CRWD", "DDOG", "OKTA",
            "ZEN", "ZM", "PINS", "SNAP", "SHOP", "BYND", "PLTR", "MDB",
            "CRSR", "SNPS", "FUBO"
        ]
        # Filter based on sector and market cap
        filtered_smallcap_tech = []
        for sym in smallcap_tech_tickers:
            if len(filtered_smallcap_tech) >= smallcap_limit:
                break
            if any(sym.endswith(suf) for suf in problematic_suffixes):
                logger.debug(f"Skipping problematic ticker: {sym}")
                continue
            try:
                info = yf.Ticker(sym).info
                if not isinstance(info, dict):
                    logger.warning(f"Info for {sym} is not a dictionary. Skipping.")
                    continue
                sector = info.get("sector", "").lower()
                market_cap = info.get("marketCap", 0)
                if sector == "technology" and market_cap <= market_cap_threshold:
                    filtered_smallcap_tech.append(sym)
                    logger.info(f"Added small-cap tech: {sym}, Market Cap=${market_cap}")
            except Exception as ee:
                logger.warning(f"Error fetching info for {sym}: {ee}")
                continue

        watchlist.update(filtered_smallcap_tech)
        logger.info(f"Fetched {len(filtered_smallcap_tech)} aggressive small-cap tech tickers.")
    except Exception as e:
        logger.error(f"Error fetching aggressive small-cap tech tickers: {e}")

    # 2. Fetch Top Daily Gainers in Tech
    try:
        # Use Yahoo Finance's daily gainers list and filter by sector
        gainers = si.get_day_gainers()
        if 'Symbol' not in gainers.columns:
            logger.error("Day gainers data does not contain 'Symbol' column. Skipping top gainers.")
        else:
            symbols = gainers['Symbol'].tolist()

            filtered_gainers = []
            for sym in symbols:
                if len(filtered_gainers) >= gainers_limit:
                    break
                if any(sym.endswith(suf) for suf in problematic_suffixes):
                    logger.debug(f"Skipping problematic ticker: {sym}")
                    continue
                try:
                    info = yf.Ticker(sym).info
                    if not isinstance(info, dict):
                        logger.warning(f"Info for {sym} is not a dictionary. Skipping.")
                        continue
                    sector = info.get("sector", "").lower()
                    if sector == "technology":
                        filtered_gainers.append(sym)
                        logger.info(f"Added top gainer tech: {sym}")
                except Exception as ee:
                    logger.warning(f"Error fetching info for {sym}: {ee}")
                    continue

            watchlist.update(filtered_gainers)
            logger.info(f"Fetched {len(filtered_gainers)} top daily gainer tech tickers.")
    except Exception as e:
        logger.error(f"Error fetching top daily gainers: {e}")

    final_watchlist = list(watchlist)
    # Remove any potential invalid tickers
    valid_watchlist = []
    for sym in final_watchlist:
        if sym in ['X']:  # Add other invalid tickers if needed
            logger.warning(f"Excluding invalid ticker: {sym}")
            continue
        valid_watchlist.append(sym)

    logger.info(f"Final watchlist => Total tickers: {len(valid_watchlist)}")
    logger.info(f"Watchlist: {valid_watchlist}")
    return valid_watchlist

###############################################################################
#                         INDICATORS & SCORING LOGIC                           #
###############################################################################
def compute_rsi(indicator: bt.ind.RSI_SMA) -> float:
    """
    Return the latest RSI value from Backtrader's RSI_SMA indicator.
    """
    return float(indicator[0])

def compute_ma_score(sma_short: float, sma_mid: float, sma_long: float) -> float:
    """
    Multi-MA: 
    - If short > mid > long, return 2
    - Else if short > mid, return 1
    - Else, return 0
    """
    if sma_short > sma_mid > sma_long:
        return 2.0
    elif sma_short > sma_mid:
        return 1.0
    return 0.0

def compute_vol_score(volume: float, vol_avg: float) -> float:
    """
    If volume > 20-day average volume, return +1; else, return 0
    """
    return 1.0 if volume > vol_avg else 0.0

def compute_atr_score(atr: float, threshold: float = 1.0) -> float:
    """
    If ATR > threshold, return +1; else, return 0
    """
    return 1.0 if atr > threshold else 0.0

def combine_scores(scores: List[float], weights: List[float]) -> float:
    """
    Combine scores with corresponding weights.
    """
    if len(scores) != len(weights):
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

###############################################################################
#                          SECTOR PERFORMANCE LOGIC                            #
###############################################################################
def fetch_tech_sector_perf(etf_symbol: str = "XLK", period="1mo") -> float:
    """
    Return the percentage change for the provided tech ETF over the specified period.
    """
    try:
        end_date = datetime.now()
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "ytd":
            start_date = datetime(year=end_date.year, month=1, day=1)
        elif period == "max":
            start_date = datetime(1900, 1, 1)
        else:
            logger.error(f"Unsupported period: {period}. Returning 0.0.")
            return 0.0

        logger.debug(f"Fetching data for {etf_symbol} from {start_date.date()} to {end_date.date()}")
        df = yf.download(
            etf_symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )

        if df.empty or len(df) < 2:
            logger.warning(f"No data for sector ETF {etf_symbol}. Returning 0.0.")
            return 0.0

        if 'Close' not in df.columns:
            logger.error(f"'Close' column not found in data for {etf_symbol}. Returning 0.0.")
            return 0.0

        # Ensure 'Close' is a Series, not a DataFrame
        if isinstance(df['Close'], pd.DataFrame):
            logger.warning(f"'Close' for {etf_symbol} returned as DataFrame. Selecting the first column.")
            df_close = df['Close'].iloc[:, 0]
        else:
            df_close = df['Close']

        start_p = float(df_close.iloc[0])
        end_p = float(df_close.iloc[-1])
        performance = (end_p - start_p) / start_p
        logger.debug(f"Sector performance for {etf_symbol}: {performance:.2%}")
        return performance
    except Exception as e:
        logger.error(f"Error fetching sector ETF {etf_symbol} performance: {e}")
        return 0.0

###############################################################################
#                           BACKTRADER STRATEGY                                #
###############################################################################
class SmallCapTechMomentum(bt.Strategy):
    """
    Momentum strategy:
    - Build watchlist from smallcap & day gainers in tech
    - Adjust thresholds by sector performance (from e.g. XLK)
    - Score: 40% MA, 30% RSI, 20% Vol, 10% ATR-based volatility
    """
    params = dict(
        buy_score_threshold=1.5,
        sell_score_threshold=0.5,
        rsi_period=14,
        atr_threshold=1.0
    )

    def __init__(self):
        self.log = logger.info
        self.live_data = False

        # Initialize indicators for each data feed
        for d in self.datas:
            d.rsi = bt.indicators.RSI_SMA(d.close, period=self.p.rsi_period)
            d.sma20 = bt.indicators.SimpleMovingAverage(d.close, period=20)
            d.sma50 = bt.indicators.SimpleMovingAverage(d.close, period=50)
            d.sma200 = bt.indicators.SimpleMovingAverage(d.close, period=200)
            d.atr = bt.indicators.ATR(d, period=14)
            d.vol_avg = bt.indicators.SimpleMovingAverage(d.volume, period=20)

    def notify_data(self, data, status, *args, **kwargs):
        dtn = data._getstatusname(status)
        self.log(f"Data {data._name} status: {dtn}")
        if dtn == "LIVE":
            self.live_data = True

    def next(self):
        # If not backtesting, wait until we have real-time data
        if not IS_BACKTEST and not self.live_data:
            return

        for d in self.datas:
            symbol = d._name
            position = self.getposition(d)

            # Ensure sufficient data for indicators
            if len(d) < 200:
                self.log(f"Not enough data for {symbol}. Skipping.")
                continue

            try:
                # Retrieve indicator values
                sma20 = d.sma20[0]
                sma50 = d.sma50[0]
                sma200 = d.sma200[0]
                rsi = d.rsi[0]
                volume = d.volume[0]
                vol_avg = d.vol_avg[0]
                atr = d.atr[0]

                # Compute scores
                ma_s = compute_ma_score(sma_short=sma20, sma_mid=sma50, sma_long=sma200)
                rsi_s = 2.0 if 40 <= rsi <= 60 else (1.0 if 30 <= rsi <= 70 else -1.0)
                vol_s = compute_vol_score(volume=volume, vol_avg=vol_avg)
                atr_s = compute_atr_score(atr=atr, threshold=self.p.atr_threshold)

                final_score = combine_scores(
                    scores=[ma_s, rsi_s, vol_s, atr_s],
                    weights=[0.4, 0.3, 0.2, 0.1]
                )

                self.log(f"{symbol} => MA={ma_s:.2f}, RSI={rsi:.1f}({rsi_s}), "
                         f"Vol={vol_s:.2f}, ATR={atr:.2f}({atr_s}), "
                         f"Final Score={final_score:.2f}")

                # BUY logic
                if position.size == 0 and final_score >= self.p.buy_score_threshold:
                    self.log(f"BUY SIGNAL for {symbol} (Score: {final_score:.2f})")
                    size_ = self.position_sizing(price=d.close[0], score=final_score)
                    if size_ > 0:
                        try:
                            self.buy(data=d, size=size_)
                            self.log(f"Placed BUY order for {size_} shares of {symbol}.")
                        except Exception as e:
                            self.log(f"Error placing BUY order for {symbol}: {e}")

                # SELL logic
                elif position.size > 0 and final_score < self.p.sell_score_threshold:
                    self.log(f"SELL SIGNAL for {symbol} (Score: {final_score:.2f})")
                    try:
                        self.close(data=d)
                        self.log(f"Placed SELL order for all shares of {symbol}.")
                    except Exception as e:
                        self.log(f"Error placing SELL order for {symbol}: {e}")

            except Exception as e:
                self.log(f"Error processing indicators for {symbol}: {e}")

    def position_sizing(self, price: float, score: float) -> int:
        """
        Basic risk-based position sizing:
        - Risk 1% of portfolio
        - Scale it by score (max 4 => scaled factor up to 1.0)
        """
        try:
            acct_value = self.broker.getvalue()
            base_risk = acct_value * 0.01
            max_score = 4.0
            scale = min(score / max_score, 1.0)
            final_risk = base_risk * (0.5 + scale)  # Ensures 0.5 * base_risk at min
            qty = int(final_risk // price)
            return max(qty, 1)
        except Exception as e:
            self.log(f"Error in position sizing: {e}")
            return 0

###############################################################################
#                                 MAIN                                        #
###############################################################################
def main():
    logger.info("=== Starting Small-Cap Tech Momentum Strategy ===")

    # 1. Build watchlist
    sc_limit = 20
    gainers_limit = 10
    watchlist = fetch_watchlist(smallcap_limit=sc_limit, gainers_limit=gainers_limit)

    if not watchlist:
        logger.info("No symbols in watchlist. Exiting.")
        return

    # 2. Compute sector performance for Tech (using XLK for 1 month)
    sector_perf = fetch_tech_sector_perf(etf_symbol="XLK", period="1mo")
    logger.info(f"Tech sector performance (1mo): {sector_perf:.2%}")

    # 3. Adjust buy/sell thresholds based on sector performance
    # Example adjustment: Increase thresholds if sector is performing well
    buy_thr = 1.5 + (sector_perf * 2.0)  # Adjust as per desired sensitivity
    sell_thr = 0.5 + (sector_perf * 1.0)

    # Ensure thresholds don't go below a minimum
    buy_thr = max(buy_thr, 1.0)
    sell_thr = max(sell_thr, 0.3)

    logger.info(f"Adjusted buy threshold: {buy_thr:.2f}, sell threshold: {sell_thr:.2f}")

    # 4. Setup Backtrader with a single AlpacaStore instance
    try:
        store = alpaca_backtrader_api.AlpacaStore(
            key_id=APCA_API_KEY_ID,
            secret_key=APCA_API_SECRET_KEY,
            paper=not IS_LIVE,  # Set to False if using live trading
        )
        broker = store.getbroker()
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            SmallCapTechMomentum,
            buy_score_threshold=buy_thr,
            sell_score_threshold=sell_thr
        )
        cerebro.setbroker(broker)
    except TypeError as e:
        logger.error(f"TypeError when setting up Alpaca broker: {e}")
        logger.error("Possible cause: Unsupported parameter 'usePolygon'. Ensure it's removed or set correctly based on your subscription.")
        sys.exit()
    except Exception as e:
        logger.error(f"Error setting up Alpaca broker: {e}")
        sys.exit()

    # 5. Add data feeds using the same AlpacaStore instance
    try:
        DataFactory = store.getdata

        if IS_BACKTEST:
            logger.info("Running in BACKTEST mode.")
            from_dt = datetime(2021, 1, 1)
            to_dt = datetime(2022, 1, 1)
            for sym in watchlist:
                try:
                    data = DataFactory(
                        dataname=sym,
                        historical=True,
                        fromdate=from_dt,
                        todate=to_dt,
                        timeframe=bt.TimeFrame.Days,
                        compression=1
                    )
                    cerebro.adddata(data, name=sym)
                    logger.info(f"Added {sym} to Backtrader data feeds (Backtest).")
                except Exception as e:
                    logger.error(f"Error adding data feed (backtest) for {sym}: {e}")
        else:
            logger.info("Running in LIVE/PAPER mode.")
            # To ensure sufficient historical data for indicators, set a fromdate
            from_dt = datetime.now() - timedelta(days=365)  # 1 year of data
            to_dt = datetime.now()
            for sym in watchlist:
                try:
                    data = DataFactory(
                        dataname=sym,
                        historical=True,  # Fetch historical data for indicators
                        fromdate=from_dt,
                        todate=to_dt,
                        timeframe=bt.TimeFrame.Days,
                        compression=1
                    )
                    cerebro.adddata(data, name=sym)
                    logger.info(f"Added {sym} to Backtrader data feeds (Live/Paper).")
                except Exception as e:
                    logger.error(f"Error adding data feed (live/paper) for {sym}: {e}")
    except Exception as e:
        logger.error(f"Error adding data feeds: {e}")
        sys.exit()

    # 6. Verify Alpaca Broker Connection and Synchronize Portfolio Value
    try:
        account = store.oapi.get_account()
        logger.info(f"Alpaca Account Status: {account.status}")
        logger.info(f"Cash: {account.cash}")
        logger.info(f"Portfolio Value: {account.portfolio_value}")

        # Remove the following lines to fix the AttributeError
        # broker.setcash(float(account.cash))
        # logger.info(f"Backtrader Broker Cash Set To: ${broker.getcash():.2f}")

        # The AlpacaBroker class already synchronizes the cash balance from Alpaca
    except Exception as e:
        logger.error(f"Error fetching Alpaca account information: {e}")
        sys.exit()

    # 7. Run
    try:
        initial_val = broker.getcash()
        logger.info(f"Initial Portfolio Value: ${initial_val:.2f}")
        cerebro.run()
        final_val = broker.getvalue()
        logger.info(f"Final Portfolio Value: ${final_val:.2f}")
    except Exception as e:
        logger.error(f"Error during Cerebro run: {e}")

if __name__ == "__main__":
    main()