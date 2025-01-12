"""
watchlist.py

Objective:
    1) Dynamically fetch a watchlist that combines:
       - US small-cap tech stocks using custom EquityQuery screener.
       - Top daily gainers in the small-cap tech sector.
       - Stocks breaking out of consolidation.
    2) Score them using a simple momentum scoring system: +1 for each day the stock has closed higher than the previous day.
    3) Ensure the watchlist maintains a minimum of 20 stocks by dynamically updating and re-populating as needed.
    4) Ensure robust logging and data handling.
"""

import os
import sys
import logging
from typing import List

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

###############################################################################
#                              ENV & LOGGING                                  #
###############################################################################
LOG_FILE = "log_watchlist.log"

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

###############################################################################
#                         WATCHLIST FETCHING LOGIC                             #
###############################################################################
def build_aggressive_small_cap_query() -> dict:
    """
    Constructs an EquityQuery dictionary to filter for aggressive small-cap US tech stocks.
    
    Returns:
        dict: The constructed EquityQuery as a dictionary.
    """
    try:
        # Define individual queries
        # Sector is Technology
        sector_eq = {
            'operator': 'eq',
            'operands': ['sector', 'Technology']
        }

        # Market Cap between 50 million USD and 2 billion USD
        market_cap_gt = {
            'operator': 'gt',
            'operands': ['intradaymarketcap', 50000000]  # Greater than 50 million
        }

        market_cap_lt = {
            'operator': 'lt',
            'operands': ['intradaymarketcap', 2000000000]  # Less than 2 billion
        }

        # Average daily volume over the last 3 months greater than 100,000
        avg_vol_gt = {
            'operator': 'gt',
            'operands': ['avgdailyvol3m', 100000]
        }

        # Combine queries using AND
        combined_query = {
            'operator': 'and',
            'operands': [sector_eq, market_cap_gt, market_cap_lt, avg_vol_gt]
        }

        return combined_query
    except Exception as e:
        logger.error(f"Error constructing Aggressive Small-Cap EquityQuery: {e}")
        sys.exit(1)

def build_top_gainers_query() -> dict:
    """
    Constructs an EquityQuery dictionary to filter for top daily gainers in small-cap tech sector.
    
    Returns:
        dict: The constructed EquityQuery as a dictionary.
    """
    try:
        # Reuse the sector and market cap criteria
        sector_eq = {
            'operator': 'eq',
            'operands': ['sector', 'Technology']
        }

        market_cap_gt = {
            'operator': 'gt',
            'operands': ['intradaymarketcap', 50000000]  # Greater than 50 million
        }

        market_cap_lt = {
            'operator': 'lt',
            'operands': ['intradaymarketcap', 2000000000]  # Less than 2 billion
        }

        # Combine queries using AND
        combined_query = {
            'operator': 'and',
            'operands': [sector_eq, market_cap_gt, market_cap_lt]
        }

        return combined_query
    except Exception as e:
        logger.error(f"Error constructing Top Gainers EquityQuery: {e}")
        sys.exit(1)

def build_breakout_query() -> dict:
    """
    Constructs an EquityQuery dictionary to filter for stocks breaking out of consolidation.
    
    Returns:
        dict: The constructed EquityQuery as a dictionary.
    """
    try:
        # Low historical volatility over last 20 days (<20)
        low_volatility = {
            'operator': 'lt',
            'operands': ['historicalvolatility', 20]
        }

        # Recent price increase above 3% in the last 3 days (adjusted from 5% to 3%)
        recent_price_spike = {
            'operator': 'gt',
            'operands': ['percentchange', 3]
        }

        # Combine queries using AND
        breakout_query = {
            'operator': 'and',
            'operands': [low_volatility, recent_price_spike]
        }

        return breakout_query
    except Exception as e:
        logger.error(f"Error constructing Breakout EquityQuery: {e}")
        sys.exit(1)

def fetch_tickers(screener: yf.Screener, screener_body: dict, limit: int) -> List[str]:
    """
    Fetches tickers based on a given screener body.
    
    Parameters:
        screener (yf.Screener): The yfinance Screener instance.
        screener_body (dict): The body of the screener query.
        limit (int): Maximum number of tickers to fetch.
    
    Returns:
        List[str]: A list of ticker symbols.
    """
    try:
        screener.set_body(screener_body)
        screener_response = screener.response

        if 'quotes' not in screener_response or not screener_response['quotes']:
            logger.warning("No stocks found for the given EquityQuery criteria.")
            return []

        tickers = [quote['symbol'] for quote in screener_response['quotes']]
        tickers = tickers[:limit]
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tickers with EquityQuery: {e}")
        return []

def fetch_watchlist(
    smallcap_limit: int = 100,
    gainers_limit: int = 100,
    breakout_limit: int = 50,
    additional_fill_limit: int = 50,  # Renamed from equityquery_fill_limit
    min_size: int = 20,
    market_cap_threshold: float = 2e9,    # 2 billion USD
    market_cap_minimum: float = 50000000  # 50 million USD
) -> List[str]:
    """
    Fetch a watchlist combining US small-cap tech stocks, top daily gainers,
    and stocks breaking out of consolidation using custom EquityQueries.
    
    Parameters:
        smallcap_limit (int): Max number of aggressive small-cap tech tickers to fetch.
        gainers_limit (int): Max number of top small-cap tech gainers to fetch.
        breakout_limit (int): Max number of stocks breaking out of consolidation to fetch.
        additional_fill_limit (int): Max number of additional small-cap tech tickers to fetch using EquityQuery.
        min_size (int): Minimum number of stocks to include in the watchlist.
        market_cap_threshold (float): Upper market cap threshold to define small-cap stocks.
        market_cap_minimum (float): Lower market cap threshold to exclude micro-cap stocks.
    
    Returns:
        List[str]: A list of unique stock ticker symbols.
    """
    watchlist = set()

    # Initialize Screener
    try:
        logger.info("Initializing yfinance Screener...")
        screener = yf.Screener()
    except AttributeError:
        logger.error("yfinance.Screener class not found. Ensure you have the latest version of yfinance installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing Screener: {e}")
        sys.exit(1)

    # Define problematic suffixes to exclude (non-US tickers)
    problematic_suffixes = [
        '.AX', '.IL', '.HK', '.BO', '.SZ', '.TA', '.SI', '.MI', '.OMX', '.L', '.A', '.B', '.C', '.D', 
        '.E', '.F', '.G', '.H', '.I', '.J', '.K', '.M', '.N', '.O', '.P', '.Q', '.R', '.S', '.T', '.U',
        '.V', '.W', '.Y', '.Z'  # Common non-US exchange suffixes
    ]

    # Helper function to validate tickers
    def validate_ticker(sym: str) -> bool:
        return not any(sym.endswith(suf) for suf in problematic_suffixes)

    # 1. Fetch Aggressive Small-Cap Tech Stocks (40% of watchlist)
    try:
        logger.info("Fetching Aggressive Small-Cap Tech Stocks using EquityQuery...")
        aggressive_query = build_aggressive_small_cap_query()

        # Screener body for Aggressive Small-Cap Tech Stocks
        aggressive_body = {
            "offset": 0,
            "size": smallcap_limit,  # Fetch up to smallcap_limit tickers
            "sortField": "intradaymarketcap",
            "sortType": "asc",  # Smaller market caps first
            "quoteType": "equity",
            "query": aggressive_query,
            "userId": "",
            "userIdType": "guid"
        }

        aggressive_tickers = fetch_tickers(screener, aggressive_body, smallcap_limit)

        # Validate and add to watchlist
        valid_aggressive_small_caps = []
        for sym in aggressive_tickers:
            if not validate_ticker(sym):
                logger.debug(f"Ticker {sym} excluded due to non-US suffix.")
                continue
            try:
                info = yf.Ticker(sym).info
                market_cap = info.get("marketCap", 0)
                if market_cap and market_cap_minimum <= market_cap < market_cap_threshold:
                    valid_aggressive_small_caps.append(sym)
                    logger.info(f"Added small-cap tech: {sym}, Market Cap=${market_cap}")
                else:
                    logger.debug(f"Ticker {sym} does not meet market cap criteria (Market Cap=${market_cap}). Skipping.")
            except Exception as ee:
                logger.warning(f"Error fetching info for {sym}: {ee}")
                continue

        watchlist.update(valid_aggressive_small_caps)
        logger.info(f"Fetched {len(valid_aggressive_small_caps)} aggressive small-cap tech tickers.")
    except Exception as e:
        logger.error(f"Error fetching aggressive small-cap tech tickers using EquityQuery: {e}")

    # 2. Fetch Top Daily Gainers in Small-Cap Tech Sector (35% of watchlist)
    try:
        logger.info("Fetching Top Daily Gainers in Small-Cap Tech Sector using EquityQuery...")
        gainers_query = build_top_gainers_query()

        # Screener body for Top Daily Gainers
        gainers_body = {
            "offset": 0,
            "size": gainers_limit,  # Fetch up to gainers_limit tickers
            "sortField": "percentchange",
            "sortType": "desc",  # Top gainers first
            "quoteType": "equity",
            "query": gainers_query,
            "userId": "",
            "userIdType": "guid"
        }

        gainers_tickers = fetch_tickers(screener, gainers_body, gainers_limit)

        # Validate and add to watchlist
        valid_small_cap_gainers = []
        for sym in gainers_tickers:
            if not validate_ticker(sym):
                logger.debug(f"Ticker {sym} excluded due to non-US suffix.")
                continue
            try:
                info = yf.Ticker(sym).info
                sector = info.get("sector", "").lower()
                market_cap = info.get("marketCap", 0)
                if sector == "technology" and market_cap and market_cap_minimum <= market_cap < market_cap_threshold:
                    valid_small_cap_gainers.append(sym)
                    logger.info(f"Added top gainer tech: {sym}, Sector={sector.capitalize()}, Market Cap=${market_cap}")
                else:
                    logger.debug(f"Ticker {sym} does not meet criteria (Sector={sector}, Market Cap=${market_cap}). Skipping.")
            except Exception as ee:
                logger.warning(f"Error fetching info for {sym}: {ee}")
                continue

        watchlist.update(valid_small_cap_gainers)
        logger.info(f"Fetched {len(valid_small_cap_gainers)} top small-cap tech gainers.")
    except Exception as e:
        logger.error(f"Error fetching small-cap tech gainers using EquityQuery: {e}")

    # 3. Fetch Stocks Breaking Out of Consolidation (15% of watchlist)
    try:
        logger.info("Fetching Stocks Breaking Out of Consolidation using EquityQuery...")
        breakout_query = build_breakout_query()

        # Screener body for Breakout Stocks
        breakout_body = {
            "offset": 0,
            "size": breakout_limit,  # Fetch up to breakout_limit tickers
            "sortField": "percentchange",
            "sortType": "desc",  # Highest percent change first
            "quoteType": "equity",
            "query": breakout_query,
            "userId": "",
            "userIdType": "guid"
        }

        breakout_tickers = fetch_tickers(screener, breakout_body, breakout_limit)

        # Validate and add to watchlist
        valid_breakout_stocks = []
        for sym in breakout_tickers:
            if not validate_ticker(sym):
                logger.debug(f"Ticker {sym} excluded due to non-US suffix.")
                continue
            try:
                info = yf.Ticker(sym).info
                market_cap = info.get("marketCap", 0)
                if market_cap and market_cap_minimum <= market_cap < market_cap_threshold:
                    valid_breakout_stocks.append(sym)
                    logger.info(f"Added breakout stock: {sym}, Market Cap=${market_cap}")
                else:
                    logger.debug(f"Ticker {sym} does not meet market cap criteria (Market Cap=${market_cap}). Skipping.")
            except Exception as ee:
                logger.warning(f"Error fetching info for breakout stock {sym}: {ee}")
                continue

        watchlist.update(valid_breakout_stocks)
        logger.info(f"Fetched {len(valid_breakout_stocks)} stocks breaking out of consolidation.")
    except Exception as e:
        logger.error(f"Error fetching breakout stocks using EquityQuery: {e}")

    # 4. Ensure Watchlist Minimum Size by Allocating Specific Counts
    # Define allocations
    allocations = {
        'aggressive_small_cap': 8,   # 40% of 20
        'top_gainers': 7,            # 35% of 20
        'breakout_stocks': 3          # 15% of 20
    }

    # Adjust allocations based on fetched tickers
    final_watchlist = set()

    # Helper function to allocate tickers
    def allocate_tickers(tickers: List[str], count: int, category: str):
        allocated = 0
        for sym in tickers:
            if sym not in final_watchlist:
                final_watchlist.add(sym)
                allocated += 1
                logger.info(f"Allocated {sym} to watchlist from {category}.")
                if allocated >= count:
                    break
        logger.info(f"Allocated {allocated} tickers from {category}.")

    # Allocate tickers per category
    # 1. Aggressive Small-Cap Tech Stocks
    allocate_tickers(valid_aggressive_small_caps, allocations['aggressive_small_cap'], "Aggressive Small-Cap Tech Stocks")

    # 2. Top Daily Gainers
    allocate_tickers(valid_small_cap_gainers, allocations['top_gainers'], "Top Daily Gainers")

    # 3. Stocks Breaking Out of Consolidation
    allocate_tickers(valid_breakout_stocks, allocations['breakout_stocks'], "Stocks Breaking Out of Consolidation")

    # Final Check
    if len(final_watchlist) < min_size:
        additional_needed = min_size - len(final_watchlist)
        logger.warning(f"After allocation, watchlist has only {len(final_watchlist)} tickers, below the minimum of {min_size}.")
        logger.info(f"Fetching additional {additional_needed} small-cap tech stocks using EquityQuery to meet the minimum watchlist size...")

        try:
            # Fetch additional Aggressive Small-Cap Tech Stocks
            additional_query = build_aggressive_small_cap_query()
            additional_body = {
                "offset": len(valid_aggressive_small_caps) + len(valid_small_cap_gainers) + len(valid_breakout_stocks),
                "size": additional_fill_limit,  # Renamed variable
                "sortField": "intradaymarketcap",
                "sortType": "asc",
                "quoteType": "equity",
                "query": additional_query,
                "userId": "",
                "userIdType": "guid"
            }

            additional_tickers = fetch_tickers(screener, additional_body, additional_fill_limit)
            valid_additional_small_caps = []
            for sym in additional_tickers:
                if not validate_ticker(sym):
                    logger.debug(f"Ticker {sym} excluded due to non-US suffix.")
                    continue
                try:
                    info = yf.Ticker(sym).info
                    market_cap = info.get("marketCap", 0)
                    if market_cap and market_cap_minimum <= market_cap < market_cap_threshold:
                        valid_additional_small_caps.append(sym)
                        logger.info(f"Added additional small-cap tech: {sym}, Market Cap=${market_cap}")
                        final_watchlist.add(sym)
                        if len(final_watchlist) >= min_size:
                            break
                    else:
                        logger.debug(f"Ticker {sym} does not meet market cap criteria (Market Cap=${market_cap}). Skipping.")
                except Exception as ee:
                    logger.warning(f"Error fetching info for {sym}: {ee}")
                    continue

            logger.info(f"Watchlist size after adding additional small-cap tech stocks: {len(final_watchlist)}")
        except Exception as e:
            logger.error(f"Error fetching additional small-cap tech tickers using EquityQuery: {e}")

    if len(final_watchlist) < min_size:
        logger.warning(f"After attempting to add additional stocks, watchlist has {len(final_watchlist)} tickers. Proceeding with available stocks.")
    else:
        logger.info(f"Watchlist meets the minimum size requirement of {min_size} stocks.")

    final_watchlist = list(final_watchlist)[:min_size]  # Limit to min_size
    logger.info(f"Final watchlist => Total tickers: {len(final_watchlist)}")
    logger.info(f"Watchlist: {final_watchlist}")

    return final_watchlist

###############################################################################
#                         SCORING LOGIC                                       #
###############################################################################
def score_watchlist(watchlist: List[str], lookback_days: int = 30) -> pd.DataFrame:
    """
    Score each stock in the watchlist based on momentum.

    Parameters:
        watchlist (List[str]): List of stock ticker symbols.
        lookback_days (int): Number of days to look back for scoring.

    Returns:
        pd.DataFrame: DataFrame containing symbols and their scores.
    """
    logger.info("Scoring watchlist based on momentum...")
    scores = []

    # Batch download historical data for efficiency
    try:
        logger.info("Downloading historical data for all tickers...")
        data = yf.download(
            tickers=watchlist,
            period="1mo",  # 1 month to cover approx 30 days
            interval="1d",
            group_by='ticker',
            threads=True,
            progress=False
        )
    except Exception as e:
        logger.error(f"Error downloading historical data: {e}")
        return pd.DataFrame()

    for sym in watchlist:
        try:
            if sym not in data.columns.levels[0]:
                logger.warning(f"No data found for {sym}. Assigning score 0.")
                scores.append({'Symbol': sym, 'Momentum_Score': 0})
                continue

            sym_data = data[sym].dropna()
            if sym_data.empty or len(sym_data) < 2:
                logger.warning(f"Not enough data to calculate momentum score for {sym}. Assigning score 0.")
                scores.append({'Symbol': sym, 'Momentum_Score': 0})
                continue

            # Calculate day-over-day changes
            df_diff = sym_data['Close'].diff().dropna()

            # Calculate the number of days with positive change
            score = (df_diff > 0).sum()

            scores.append({'Symbol': sym, 'Momentum_Score': score})
            logger.info(f"Scored {sym}: {score}")
        except Exception as ee:
            logger.error(f"Error processing {sym}: {ee}")
            scores.append({'Symbol': sym, 'Momentum_Score': 0})
            continue

    score_df = pd.DataFrame(scores)
    score_df.sort_values(by='Momentum_Score', ascending=False, inplace=True)
    logger.info("Scoring completed.")
    logger.debug(score_df)
    return score_df

###############################################################################
#                         FILTERING WATCHLIST                                 #
###############################################################################
def filter_watchlist(scored_df: pd.DataFrame, min_size: int = 20) -> pd.DataFrame:
    """
    Filter the watchlist to retain the top tickers based on momentum scores,
    ensuring the watchlist meets the minimum size requirement.

    Parameters:
        scored_df (pd.DataFrame): DataFrame containing symbols and their scores.
        min_size (int): Minimum number of tickers to retain in the watchlist.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    logger.info("Filtering watchlist based on momentum scores...")

    if scored_df.empty:
        logger.warning("Scored DataFrame is empty. Returning empty DataFrame.")
        return scored_df

    # Determine the number of tickers to retain
    retain_count = max(min_size, len(scored_df) // 2)
    if retain_count > len(scored_df):
        retain_count = len(scored_df)

    filtered_df = scored_df.head(retain_count)
    logger.info(f"Filtered watchlist to top {retain_count} tickers.")
    logger.debug(filtered_df)
    return filtered_df

###############################################################################
#                                 MAIN                                        #
###############################################################################
def main():
    logger.info("=== Starting Momentum Strategy Watchlist Fetching ===")

    # 1. Build watchlist
    sc_limit = 100
    gainers_limit = 100
    breakout_limit = 50
    additional_fill_limit = 50  # Renamed from equityquery_fill_limit
    min_size = 20
    market_cap_threshold = 2e9      # 2 billion USD
    market_cap_minimum = 50000000   # 50 million USD

    watchlist = fetch_watchlist(
        smallcap_limit=sc_limit, 
        gainers_limit=gainers_limit,
        breakout_limit=breakout_limit,
        additional_fill_limit=additional_fill_limit,
        min_size=min_size,
        market_cap_threshold=market_cap_threshold,
        market_cap_minimum=market_cap_minimum
    )

    if not watchlist:
        logger.info("No symbols in watchlist. Exiting.")
        return

    # 2. Score watchlist
    lookback_days = 30  # Number of days to look back for momentum
    scored_df = score_watchlist(watchlist, lookback_days)

    if scored_df.empty:
        logger.info("No scores calculated. Exiting.")
        return

    # 3. Filter watchlist
    filtered_df = filter_watchlist(scored_df, min_size=min_size)

    if filtered_df.empty:
        logger.info("No symbols after filtering. Exiting.")
        return

    # 4. Display final watchlist
    logger.info("Final Watchlist after Scoring and Filtering:")
    logger.info(filtered_df.to_string(index=False))

    # Optional: Save to CSV
    try:
        filtered_df.to_csv("final_watchlist.csv", index=False)
        logger.info("Final watchlist saved to 'final_watchlist.csv'.")
    except Exception as e:
        logger.error(f"Error saving final watchlist to CSV: {e}")

if __name__ == "__main__":
    main()
