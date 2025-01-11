"""
equitytest.py

Objective:
    1) Dynamically fetch a watchlist that combines:
       - US small-cap tech stocks using EquityQuery screener.
    2) Score them using a simple momentum scoring system: +1 for each day the stock has closed higher than the previous day.
    3) Ensure the watchlist maintains a minimum of 20 stocks by dynamically updating and re-populating as needed.
    4) Ensure robust logging and data handling.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List

import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

###############################################################################
#                              ENV & LOGGING                                  #
###############################################################################
LOG_FILE = "log_equity.log"

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
def build_equity_query() -> yf.EquityQuery:
    """
    Constructs an EquityQuery to filter for promising small-cap US tech stocks suitable for a momentum strategy.

    Returns:
        yf.EquityQuery: The constructed EquityQuery object.
    """
    try:
        # Define individual queries
        # Sector is Technology
        sector_eq = yf.EquityQuery('eq', ['sector', 'Technology'])

        # Market Cap less than 2 billion USD
        market_cap_lt = yf.EquityQuery('lt', ['intradaymarketcap', 2e9])

        # Average daily volume over the last 3 months greater than 100,000
        avg_vol_gt = yf.EquityQuery('gt', ['avgdailyvol3m', 100000])

        # Combine queries using AND
        combined_query = yf.EquityQuery('and', [sector_eq, market_cap_lt, avg_vol_gt])

        return combined_query
    except Exception as e:
        logger.error(f"Error constructing EquityQuery: {e}")
        sys.exit(1)

def fetch_watchlist(
    min_size: int = 20
) -> List[str]:
    """
    Fetch a watchlist of small-cap US tech stocks using EquityQuery.

    Parameters:
        min_size (int): Minimum number of stocks to include in the watchlist.

    Returns:
        List[str]: A list of unique stock ticker symbols.
    """
    watchlist = set()

    try:
        logger.info("Building EquityQuery for small-cap US Technology stocks...")
        equity_query = build_equity_query()

        logger.info("Executing EquityQuery...")
        screener = yf.Screener()

        # Build the screener body
        body = {
            "offset": 0,
            "size": 100,  # Maximum allowed by Yahoo is 250
            "sortField": "eodvolume",
            "sortType": "desc",
            "quoteType": "equity",
            "query": equity_query.to_dict(),
            "userId": "",
            "userIdType": "guid"
        }

        screener.set_body(body)
        screener_response = screener.response

        if 'quotes' not in screener_response or not screener_response['quotes']:
            logger.warning("No stocks found with the initial EquityQuery criteria.")
        else:
            initial_stocks = [quote['symbol'] for quote in screener_response['quotes']]
            watchlist.update(initial_stocks)
            logger.info(f"Fetched {len(initial_stocks)} stocks from EquityQuery.")
    except Exception as e:
        logger.error(f"Error fetching watchlist using EquityQuery: {e}")

    # If watchlist is smaller than min_size, relax the criteria
    if len(watchlist) < min_size:
        logger.warning(f"Watchlist has only {len(watchlist)} tickers, which is below the minimum of {min_size}.")
        additional_needed = min_size - len(watchlist)
        logger.info(f"Relaxing EquityQuery criteria to fetch additional {additional_needed} stocks.")

        try:
            # Relax average daily volume to >50,000
            avg_vol_relaxed = yf.EquityQuery('gt', ['avgdailyvol3m', 50000])

            # Combine with sector and market cap
            relaxed_query = yf.EquityQuery('and', [
                yf.EquityQuery('eq', ['sector', 'Technology']),
                yf.EquityQuery('lt', ['intradaymarketcap', 2e9]),
                avg_vol_relaxed
            ])

            logger.info("Executing relaxed EquityQuery...")

            # Build the relaxed screener body
            relaxed_body = {
                "offset": 0,
                "size": 100,  # Maximum allowed by Yahoo is 250
                "sortField": "eodvolume",
                "sortType": "desc",
                "quoteType": "equity",
                "query": relaxed_query.to_dict(),
                "userId": "",
                "userIdType": "guid"
            }

            screener.set_body(relaxed_body)
            screener_response = screener.response

            if 'quotes' not in screener_response or not screener_response['quotes']:
                logger.warning("No stocks found with the relaxed EquityQuery criteria.")
            else:
                additional_stocks = [quote['symbol'] for quote in screener_response['quotes']]
                watchlist.update(additional_stocks)
                logger.info(f"Fetched {len(additional_stocks)} additional stocks from relaxed EquityQuery.")
        except Exception as e:
            logger.error(f"Error fetching additional watchlist using relaxed EquityQuery: {e}")

    # Final check to ensure min_size
    if len(watchlist) < min_size:
        logger.warning(f"After relaxing criteria, watchlist has {len(watchlist)} tickers. Proceeding with available stocks.")
    else:
        logger.info(f"Watchlist meets the minimum size requirement of {min_size} stocks.")

    final_watchlist = list(watchlist)[:min_size]  # Limit to min_size
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
    watchlist = fetch_watchlist(min_size=20)

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
    MIN_WATCHLIST_SIZE = 20
    filtered_df = filter_watchlist(scored_df, min_size=MIN_WATCHLIST_SIZE)

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
