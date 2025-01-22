"""
live_momentum_bot.py

A live-trading bot on Alpaca's paper environment that:
1) Dynamically fetches a watchlist from:
    - 5 daily US tech gainers (above $3, small-cap <2B, volume>200k),
    - 25 additional small-cap US tech stocks with medium/high volume,
2) Downloads historical bar data via the Alpaca Market Data HTTP API (IEX feed),
3) Uses advanced multi-indicator "final_score_indicators" (MACD, RSI, RSI_STD, ATR, SECTOR, RVOL, ADX_DI, CROSSOVER),
4) Places limit buy orders if the score is above BUY_SCORE_THRESHOLD,
5) Uses ATR-based stop-loss and profit target,
6) Closes positions near EOD or on triggers,
7) Logs all trades to CSV.

Dependencies:
- alpaca_trade_api
- alpaca-py (pip install alpaca-py)
- python-dotenv
- yahoo_fin
- pandas, numpy, yfinance
- ta (optional, if you'd like more advanced indicators)

This code is for demonstration and educational purposes. 
Use caution in a production environment.
"""

import os
import sys
import time
import json
import logging
import asyncio
import signal
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, List, Optional

# ------------------ MARKET DATA SDK IMPORTS ------------------
# Install with: pip install alpaca-py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
# -----------------------------------------------------------

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

# Yahoo_fin for watchlist scraping
try:
    from yahoo_fin import stock_info as si
except ImportError:
    print("Please install yahoo_fin via 'pip install yahoo_fin'")
    sys.exit(1)

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

###############################################################################
#                          LOGGING CONFIGURATION                              #
###############################################################################
LOG_LEVEL = logging.DEBUG
logger = logging.getLogger("LiveMomentumBot")
logger.setLevel(LOG_LEVEL)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
fh = logging.FileHandler("live_momentum_bot.log")
fh.setLevel(LOG_LEVEL)
fh.setFormatter(formatter)
logger.addHandler(fh)

# CSV file for trade logs
TRADE_LOG_FILE = "live_momentum_trades.csv"
if not os.path.exists(TRADE_LOG_FILE):
    with open(TRADE_LOG_FILE, "w") as f:
        f.write("TimeUTC,Symbol,Side,Qty,Price,Reason,PnL,IsDayTrade\n")

###############################################################################
#                          ENV & ALPACA API INIT                              #
###############################################################################
# Load environment variables
load_dotenv("env.txt")

# Fetch Alpaca API credentials and configuration
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_PAPER = True  # Paper trading mode (True for sandbox, False for live trading)

# REST API Base URL
APCA_API_BASE_URL = "https://paper-api.alpaca.markets" if APCA_PAPER else "https://api.alpaca.markets"

# Ensure API credentials are set
if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    logger.error("Missing Alpaca credentials. Please set them in your .env file.")
    sys.exit(1)

# Initialize Alpaca REST API
try:
    alpaca = tradeapi.REST(
        key_id=APCA_API_KEY_ID,
        secret_key=APCA_API_SECRET_KEY,
        base_url=APCA_API_BASE_URL,
        api_version="v2"
    )
    account_info = alpaca.get_account()
    logger.info(f"Connected to Alpaca. Account status={account_info.status}, Equity={account_info.equity}")
except Exception as e:
    logger.error(f"Error connecting to Alpaca REST API: {e}")
    sys.exit(1)

# Initialize the Alpaca Market Data HTTP client
stock_data_client = StockHistoricalDataClient(
    api_key=APCA_API_KEY_ID,
    secret_key=APCA_API_SECRET_KEY
)

###############################################################################
#                       GLOBAL SETTINGS & CONSTANTS                           #
###############################################################################
MAX_WATCHLIST_SIZE = 30
FETCH_INTERVAL_MINUTES = 5
BUY_SCORE_THRESHOLD = 0.7
PDT_ENABLED = True
MINUTES_BEFORE_CLOSE_TO_EXIT = 3  # Flatten positions near EOD
ALLOC_PCT = 0.07  # 7% of equity per position
MAX_DAY_TRADES_5D = 3

STOP_LOSS_MULT = 1.5
PROFIT_TARGET_MULT = 3.0

# Weighted indicators
WEIGHTS_DICT = {
    "rsi": 0.12442881613969803,
    "rsi_std": 0.12087585777044296,
    "macd": 0.13195861876010895,
    "atr_filter": 0.11658483743667603,
    "sector": 0.11400578916072845,
    "rvol": 0.13725993037223816,
    "adx_di": 0.12422795593738556,
    "crossover": 0.13065817952156067
}
VOLATILITY_THRESHOLD = 1.0

###############################################################################
#                   UTILITY FUNCTIONS (Parsing, etc.)                        #
###############################################################################
def parse_price(val) -> float:
    try:
        if isinstance(val, str):
            val = val.replace('$', '').replace(',', '').strip()
        parsed_price = float(val)
        return parsed_price if parsed_price > 0 else 0.0
    except Exception:
        return 0.0

def parse_volume(val) -> float:
    try:
        if isinstance(val, str):
            val = val.lower().replace(',', '')
            if 'b' in val:
                num = float(val.replace('b', ''))
                return num * 1_000_000_000
            elif 'm' in val:
                num = float(val.replace('m', ''))
                return num * 1_000_000
            elif 'k' in val:
                num = float(val.replace('k', ''))
                return num * 1_000
            return float(val)
        return float(val)
    except Exception:
        return 0.0

def parse_market_cap(val) -> float:
    return parse_volume(val)

def is_us_stock(sym: str) -> bool:
    foreign_suffixes = [
        ".AR", ".AT", ".AU", ".BE", ".BR", ".CA", ".CH", ".CL", ".CN", ".CZ",
        ".DE", ".DK", ".EE", ".EG", ".ES", ".FI", ".FR", ".GB", ".GR", ".HK",
        ".HU", ".ID", ".IE", ".IL", ".IN", ".IS", ".IT", ".JP", ".KR", ".KW",
        ".LK", ".LT", ".LV", ".MX", ".MY", ".NL", ".NO", ".NZ", ".PE", ".PH",
        ".PK", ".PL", ".PT", ".QA", ".RO", ".RU", ".SA", ".SE", ".SG", ".SR",
        ".TH", ".TR", ".TW", ".VE", ".VN", ".ZA", ".OL", ".PA", ".TO", ".BK",
        ".L", ".F", ".V", ".NS", ".T", ".SR", ".CR"
    ]
    sym_up = sym.upper()
    return not any(sym_up.endswith(suf) for suf in foreign_suffixes)

###############################################################################
#    QUERY CONSTRUCTION FUNCTIONS FOR SMALL CAP TECH STOCKS                #
###############################################################################
def build_small_cap_query(lower_bound: float, upper_bound: float) -> dict:
    try:
        sector_eq = {'operator': 'eq', 'operands': ['sector', 'Technology']}
        region_eq = {'operator': 'eq', 'operands': ['region', 'us']}
        market_cap_gt = {'operator': 'gt', 'operands': ['intradaymarketcap', lower_bound]}
        market_cap_lt = {'operator': 'lt', 'operands': ['intradaymarketcap', upper_bound]}
        avg_vol_gt = {'operator': 'gt', 'operands': ['avgdailyvol3m', 100_000]}
        query = {
            'operator': 'and',
            'operands': [sector_eq, region_eq, market_cap_gt, market_cap_lt, avg_vol_gt]
        }
        logger.debug(f"Built small cap query for market cap range: ${lower_bound} - ${upper_bound}")
        return query
    except Exception as e:
        logger.error(f"Error constructing small cap query: {e}")
        sys.exit(1)

def fetch_tickers_for_query(query: dict, fetch_size: int = 250) -> List[str]:
    try:
        screener = yf.Screener()
        screener_body = {
            "offset": 0,
            "size": fetch_size,
            "sortField": "intradaymarketcap",
            "sortType": "asc",
            "quoteType": "equity",
            "query": query,
            "userId": "",
            "userIdType": "guid"
        }
        screener.set_body(screener_body)
        response = screener.response
        if 'quotes' not in response or not response['quotes']:
            logger.warning("No quotes returned for the given query.")
            return []
        tickers = [quote['symbol'] for quote in response['quotes']]
        logger.debug(f"Fetched tickers from query: {tickers}")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tickers with query: {e}")
        return []

###############################################################################
#                          FETCH DYNAMIC WATCHLIST                            #
###############################################################################
def fetch_dynamic_watchlist() -> List[str]:
    desired_count = 30
    volume_threshold = 200_000
    final_list = []
    query = build_small_cap_query(50_000_000, 2_000_000_000)
    candidates = fetch_tickers_for_query(query, fetch_size=250)
    logger.info(f"Candidates fetched from screener: {len(candidates)}")
    for ticker in candidates:
        logger.debug(f"Processing ticker: {ticker}")
        if not is_us_stock(ticker):
            logger.debug(f"Ticker {ticker} rejected due to foreign suffix.")
            continue
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = parse_price(info.get("regularMarketPrice", 0))
            if price == 0.0:
                try:
                    hist_data = stock.history(period="1d", interval="1d")
                    price = hist_data['Close'].iloc[-1] if not hist_data.empty else 0.0
                except Exception as e:
                    logger.warning(f"Error fetching historical price for {ticker}: {e}")
                    price = 0.0
            volume = parse_volume(info.get("regularMarketVolume", 0))
            market_cap = parse_market_cap(info.get("marketCap", 0))
            sector = info.get("sector", "").lower()
            logger.debug(f"Ticker={ticker}, Price={price}, Volume={volume}, MarketCap={market_cap}, Sector={sector}")
            if (price > 3 and volume >= volume_threshold and 
                50_000_000 <= market_cap < 2_000_000_000 and 
                "tech" in sector):
                final_list.append(ticker)
                logger.info(f"Ticker {ticker} accepted: Price=${price}, Volume={volume}, Market Cap=${market_cap}")
            else:
                logger.debug(f"Ticker {ticker} rejected based on values.")
            if len(final_list) >= desired_count:
                break
        except Exception as e:
            logger.warning(f"Error processing ticker {ticker}: {e}")
    logger.info(f"Dynamic watchlist constructed: {final_list}")
    return final_list

###############################################################################
#                              ROTATE WATCHLIST                             #
###############################################################################
scoreboard_for_rotation = {}

def rotate_watchlist(current_watchlist: List[str]) -> List[str]:
    global scoreboard_for_rotation
    if not scoreboard_for_rotation:
        logger.info("No previous scoreboard available; fetching fresh watchlist.")
        return fetch_dynamic_watchlist()
    top_count = MAX_WATCHLIST_SIZE // 2
    sorted_scores = sorted(scoreboard_for_rotation.items(), key=lambda x: x[1], reverse=True)
    top_half = [ticker for (ticker, score) in sorted_scores if ticker in current_watchlist][:top_count]
    logger.debug(f"Preserving top half tickers: {top_half}")
    new_candidates = fetch_dynamic_watchlist()
    new_list = list(top_half)
    for ticker in new_candidates:
        if ticker not in new_list:
            new_list.append(ticker)
        if len(new_list) >= MAX_WATCHLIST_SIZE:
            break
    logger.info(f"Rotated watchlist: {new_list}")
    return new_list

###############################################################################
#                      DATA & POSITIONS & PDT TRACKING                         #
###############################################################################
symbol_bars: Dict[str, pd.DataFrame] = {}
open_positions: Dict[str, Dict[str, Any]] = {}
day_trade_timestamps: List[datetime] = []

###############################################################################
#                        ADVANCED SCORING INDICATORS                           #
###############################################################################
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

def compute_percentile_rank(series: pd.Series, value: float) -> float:
    if len(series) < 2:
        return 0.5
    return (series < value).mean()

def compute_rsi_score(rsi_series: pd.Series, rsi_volatility: Optional[float] = None, sector_factor: float = 0.0) -> float:
    if rsi_series.empty:
        return 0.5
    if rsi_volatility is None:
        lookback_window = 14
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

def compute_macd_score(prices: pd.Series) -> float:
    macd_line, signal_line, hist = compute_macd(prices)
    if len(hist) == 0:
        return 0.5
    latest_hist = hist.iloc[-1]
    base_score = 0.7 if latest_hist > 0 else 0.3
    if len(hist) > 1:
        prev_hist = hist.iloc[-2]
        if prev_hist < 0 < latest_hist:
            base_score += 0.2
        elif prev_hist > 0 > latest_hist:
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

def compute_atr_filter_score(df: pd.DataFrame, period=14) -> float:
    atr_series = compute_atr(df, period=period)
    if len(atr_series) < 1:
        return 0.5
    curr_atr = atr_series.iloc[-1]
    if curr_atr < 1.0:
        return 0.8
    elif curr_atr > 5.0:
        return 0.2
    else:
        return 0.5

def compute_adx_di_score(df: pd.DataFrame, period: int = 14) -> float:
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

def compute_crossover_score(df: pd.DataFrame, short_period=10, long_period=30) -> float:
    if len(df) < long_period:
        return 0.5
    sma_short = df["Close"].rolling(short_period).mean().iloc[-1]
    sma_long = df["Close"].rolling(long_period).mean().iloc[-1]
    diff = sma_short - sma_long
    return 1.0 if diff > 0 else 0.0

def compute_relative_volume_score(df: pd.DataFrame, short_window=5, long_window=20) -> float:
    if len(df) < long_window:
        return 0.5
    short_avg = df["Volume"].tail(short_window).mean()
    long_avg = df["Volume"].tail(long_window).mean()
    if long_avg <= 0:
        return 0.5
    rvol = short_avg / long_avg
    if rvol >= 2.0:
        return 1.0
    elif rvol >= 1.5:
        return 0.7
    elif rvol >= 1.0:
        return 0.5
    else:
        return 0.3

# We'll use a sector ETF (XLK) for "sector" performance.
SECTOR_ETF = "XLK"
SECTOR_DF = pd.DataFrame()

def fetch_sector_data(etf_symbol: str, lookback_days=90) -> pd.DataFrame:
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    try:
        df = yf.download(etf_symbol,
                         start=start_dt.strftime("%Y-%m-%d"),
                         end=end_dt.strftime("%Y-%m-%d"),
                         interval="1d", progress=False)
        return df
    except Exception as e:
        logger.error(f"Error fetching sector data for {etf_symbol}: {e}")
        return pd.DataFrame()

def compute_sector_factor(signal_date: pd.Timestamp, rolling_window=5) -> float:
    global SECTOR_DF
    df = SECTOR_DF
    if df.empty or "Close" not in df.columns:
        return 0.0
    if signal_date not in df.index:
        # Find the nearest preceding date
        possible = df.index[df.index <= signal_date]
        if len(possible) == 0:
            return 0.0
        signal_date = possible[-1]
    pos = df.index.get_loc(signal_date)
    start_pos = max(0, pos - rolling_window + 1)
    segment = df["Close"].iloc[start_pos:pos+1]
    if len(segment) < 2:
        return 0.0
    first_p = segment.iloc[0]
    last_p = segment.iloc[-1]
    if first_p == 0:
        return 0.0
    base_perf = (last_p - first_p) / first_p
    return float(np.clip(base_perf * 10, -2, 2))

def final_score_indicators(df: pd.DataFrame) -> float:
    if len(df) < 30:
        return 0.0
    signal_date = df.index[-1]
    sector_score_factor = compute_sector_factor(signal_date, rolling_window=5)
    rsi_series = compute_rsi(df["Close"], 14)
    rsi_s = compute_rsi_score(rsi_series, sector_factor=sector_score_factor)
    rsi_std_s = compute_rsi_std_score(rsi_series, 14, 60, market_trend=sector_score_factor)
    macd_s = compute_macd_score(df["Close"])
    atr_f = compute_atr_filter_score(df, 14)
    sec_map = (sector_score_factor + 2.0) / 4.0
    sec_map = float(np.clip(sec_map, 0.0, 1.0))
    rvol_s = compute_relative_volume_score(df, 5, 20)
    adx_s = compute_adx_di_score(df)
    cross_s = compute_crossover_score(df, 10, 30)
    indicators = {
        "rsi": rsi_s,
        "rsi_std": rsi_std_s,
        "macd": macd_s,
        "atr_filter": atr_f,
        "sector": sec_map,
        "rvol": rvol_s,
        "adx_di": adx_s,
        "crossover": cross_s
    }
    ws = WEIGHTS_DICT
    weighted_sum = sum(indicators[k] * ws.get(k, 0.0) for k in indicators)
    total_weight = sum(ws.get(k, 0.0) for k in indicators)
    if total_weight == 0:
        return 0.5
    raw_score = weighted_sum / total_weight
    logger.debug(f"Indicators: RSI={rsi_s:.2f}, MACD={macd_s:.2f}, ATR={atr_f:.2f}, Sector={sec_map:.2f}, rvol={rvol_s:.2f}, adx={adx_s:.2f}, crossover={cross_s:.2f}")
    if atr_f < 0.5:
        raw_score -= 0.1 * VOLATILITY_THRESHOLD
    return float(np.clip(raw_score, 0.0, 1.0))

###############################################################################
#                             ORDER & PDT LOGIC                              #
###############################################################################
def can_open_new_position(symbol: str) -> bool:
    if symbol in open_positions:
        return False
    try:
        orders = alpaca.list_orders(status="open", symbol=symbol)
        if orders:
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking open orders for {symbol}: {e}")
        return False

def check_pdt_capacity() -> bool:
    if not PDT_ENABLED:
        return True
    now = datetime.utcnow()
    five_days_ago = now - timedelta(days=5)
    try:
        trades = alpaca.get_activities(activity_types='FILL', after=five_days_ago.isoformat())
        trades_by_symbol_day = defaultdict(list)
        for t in trades:
            if t.symbol and t.transaction_time:
                trade_day = pd.to_datetime(t.transaction_time).date()
                trades_by_symbol_day[(t.symbol, trade_day)].append(t)
        day_trades_count = 0
        for (symbol, d), tlist in trades_by_symbol_day.items():
            sides = set([x.side.lower() for x in tlist])
            if "buy" in sides and "sell" in sides:
                day_trades_count += 1
        if day_trades_count >= MAX_DAY_TRADES_5D:
            return False
    except Exception as e:
        logger.error(f"Error fetching day trades from Alpaca: {e}")
        count_dt = sum(1 for dt in day_trade_timestamps if dt >= five_days_ago)
        return count_dt < MAX_DAY_TRADES_5D
    return True

def place_limit_buy(symbol: str, score: float, latest_price: float):
    try:
        account = alpaca.get_account()
        equity = float(account.equity)
    except Exception as e:
        logger.error(f"Error fetching account for sizing: {e}")
        return
    base_amount = equity * ALLOC_PCT
    scale_factor = 0.5 * score + 0.5
    position_size_value = base_amount * scale_factor
    qty = int(position_size_value // latest_price)
    if qty < 1:
        logger.info(f"Position sizing => 0 shares for {symbol}. Skip.")
        return
    stop_level = latest_price * (1 - 0.02 * STOP_LOSS_MULT)
    target_level = latest_price * (1 + 0.02 * PROFIT_TARGET_MULT)
    limit_price = round(latest_price * 1.001, 2)
    logger.info(f"Placing LIMIT BUY for {symbol}, {qty} shares ~{limit_price:.2f}")
    success = False
    for attempt in range(3):
        try:
            alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                type="limit",
                time_in_force="day",
                limit_price=str(limit_price),
                order_class="bracket",
                stop_loss={"stop_price": str(stop_level)},
                take_profit={"limit_price": str(target_level)}
            )
            success = True
            break
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Error placing limit buy for {symbol}: {e}")
            time.sleep(1)
    if not success:
        logger.error(f"Failed to place limit buy for {symbol} after 3 attempts.")
        return
    open_positions[symbol] = {
        "qty": qty,
        "entry_price": latest_price,
        "score": score,
        "stop_price": stop_level,
        "target_price": target_level,
        "opened_at": datetime.utcnow()
    }

def execute_close(symbol: str, reason: str, px: float):
    pos = open_positions[symbol]
    qty = pos["qty"]
    try:
        open_orders = alpaca.list_orders(status="open", symbol=symbol)
        for o in open_orders:
            for attempt in range(3):
                try:
                    alpaca.cancel_order(o.id)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt+1}: Error canceling open order for {symbol}: {e}")
                    time.sleep(1)
    except Exception as e:
        logger.error(f"Error fetching open orders for {symbol}: {e}")
    sell_success = False
    for attempt in range(3):
        try:
            alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day"
            )
            sell_success = True
            break
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Error placing sell for {symbol}: {e}")
            time.sleep(1)
    if not sell_success:
        logger.error(f"Failed to place sell order for {symbol} after 3 attempts.")
        return
    entry_px = pos["entry_price"]
    pnl = (px - entry_px) * qty
    is_day_trade = (pos["opened_at"].date() == datetime.utcnow().date())
    if is_day_trade:
        day_trade_timestamps.append(datetime.utcnow())
    with open(TRADE_LOG_FILE, "a") as f:
        f.write(f"{datetime.utcnow()},{symbol},sell,{qty},{px:.2f},{reason},{pnl:.2f},{is_day_trade}\n")
    logger.info(f"Closed {symbol} reason={reason}, PnL={pnl:.2f}, dayTrade={is_day_trade}")

###############################################################################
#                  STOP/TARGET DYNAMIC UPDATE LOGIC                          #
###############################################################################
def update_stop_target_prices(symbol: str, current_price: float):
    pos = open_positions[symbol]
    new_stop = max(pos['stop_price'], current_price * (1 - 0.02 * STOP_LOSS_MULT))
    pos['stop_price'] = new_stop
    new_target = max(pos['target_price'], current_price * (1 + 0.02 * PROFIT_TARGET_MULT))
    pos['target_price'] = new_target

###############################################################################
#             CHECK POSITION EXITS (NEWLY ADDED FUNCTION)                    #
###############################################################################
def check_position_exits():
    to_remove = []
    now_utc = datetime.utcnow()
    close_utc = now_utc.replace(hour=20, minute=0, second=0, microsecond=0)
    flatten_now = False
    if (close_utc - now_utc).total_seconds() <= (MINUTES_BEFORE_CLOSE_TO_EXIT * 60):
        flatten_now = True
    for sym, pos in list(open_positions.items()):
        df = symbol_bars.get(sym, None)
        if df is None or len(df) == 0:
            continue
        current_price = df["Close"].iloc[-1]
        update_stop_target_prices(sym, current_price)
        if current_price <= pos["stop_price"]:
            logger.info(f"{sym} hit STOP at price {current_price:.2f}")
            execute_close(sym, "stop-hit", current_price)
            to_remove.append(sym)
        elif current_price >= pos["target_price"]:
            logger.info(f"{sym} hit TARGET at price {current_price:.2f}")
            execute_close(sym, "target-hit", current_price)
            to_remove.append(sym)
        elif flatten_now:
            logger.info(f"{sym} flatten EOD at price {current_price:.2f}")
            execute_close(sym, "eod-close", current_price)
            to_remove.append(sym)
    for sym in to_remove:
        if sym in open_positions:
            del open_positions[sym]

###############################################################################
#         PERIODIC TASKS & MARKET DATA RETRIEVAL (UPDATED)                   #
###############################################################################
last_sector_update = None

async def periodic_tasks():
    global watchlist, scoreboard_for_rotation, SECTOR_DF, last_sector_update

    # 1) Refresh sector data (once a day)
    if last_sector_update is None or (datetime.utcnow() - last_sector_update).days >= 1:
        logger.info("Refreshing sector data...")
        SECTOR_DF = fetch_sector_data(SECTOR_ETF, 90)
        last_sector_update = datetime.utcnow()

    # 2) Update watchlist rotation every hour (if needed)
    now = datetime.utcnow()
    if now.minute == 0:
        watchlist = rotate_watchlist(watchlist)
        logger.info(f"Rotated watchlist => {watchlist}")

    # 3) MARKET DATA API CHANGE:
    # Fetch the last 60 minutes of 1-minute bars from Alpaca using free IEX data.
    # Use end time = current time minus 15 minutes (due to IEX delay) and start time = 60 minutes before that.
    end_utc = datetime.utcnow() - timedelta(minutes=15)
    start_utc = end_utc - timedelta(minutes=60)
    request_params = StockBarsRequest(
        symbol_or_symbols=watchlist,
        timeframe=TimeFrame.Minute,
        start=start_utc,
        end=end_utc,
        feed="iex"  # Explicitly set the feed to IEX
    )
    try:
        bars_response = stock_data_client.get_stock_bars(request_params)
        all_bars_df = bars_response.df
        if all_bars_df is not None and not all_bars_df.empty:
            for sym in watchlist:
                if sym in all_bars_df.index.get_level_values("symbol"):
                    sym_bars = all_bars_df.xs(sym, level="symbol").reset_index()
                    sym_bars.rename(columns={
                        "timestamp": "timestamp",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume"
                    }, inplace=True)
                    sym_bars.set_index("timestamp", inplace=True)
                    symbol_bars[sym] = sym_bars
                else:
                    logger.warning(f"No IEX data available for symbol: {sym}")
        else:
            logger.warning("No bar data returned for watchlist.")
    except Exception as e:
        logger.error(f"Error fetching 60-minute bars from Alpaca: {e}")

    # 4) Score each symbol
    scoreboard = []
    for sym in watchlist:
        df = symbol_bars.get(sym, None)
        if df is None or len(df) < 30:
            scoreboard.append((sym, 0.0))
        else:
            score_val = final_score_indicators(df)
            scoreboard.append((sym, score_val))
    scoreboard.sort(key=lambda x: x[1], reverse=True)
    scoreboard_for_rotation = dict(scoreboard)
    logger.info(f"Top 5 scoreboard => {scoreboard[:5]}")

    # 5) Attempt new buys
    for sym, sc in scoreboard:
        if sc >= BUY_SCORE_THRESHOLD:
            if can_open_new_position(sym) and check_pdt_capacity():
                df = symbol_bars[sym]
                last_close = df["Close"].iloc[-1]
                place_limit_buy(sym, sc, last_close)

    # 6) Check stops/targets
    check_position_exits()

    # 7) Trim symbol_bars to last 1 day (to prevent memory bloat)
    for s in symbol_bars:
        symbol_bars[s] = symbol_bars[s].last("1D")

async def run_scheduled():
    while True:
        try:
            await periodic_tasks()
        except Exception as e:
            logger.error(f"Error in periodic tasks: {e}")
        await asyncio.sleep(FETCH_INTERVAL_MINUTES * 60)

###############################################################################
#                            SCRIPT ENTRY POINT                              #
###############################################################################
if __name__ == "__main__":
    # 1) Fetch sector data at startup
    SECTOR_DF = fetch_sector_data(SECTOR_ETF, 90)
    last_sector_update = datetime.utcnow()

    # 2) Build initial watchlist
    watchlist = fetch_dynamic_watchlist()
    logger.info(f"Initial watchlist => {watchlist}")

    # 3) Setup graceful termination signals
    def signal_handler(sig, frame):
        logger.info(f"Received termination signal {sig}. Exiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 4) Start periodic tasks using asyncio (no WebSocket stream)
    try:
        asyncio.run(run_scheduled())
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
