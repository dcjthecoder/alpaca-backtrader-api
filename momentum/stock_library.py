"""
stock_library.py

Objective:
    Create a library of US Small Cap Tech Stocks that:
      - Have valid US tickers.
      - Possess historical data available from 1/1/2019 to present.
      - Have experienced at least a 10% price change since 1/1/2019.
    The library will contain at least 500 tickers stored in a CSV file,
    with one ticker per row (position-based selection is easily performed).
"""

import os
import sys
import logging
import time
from typing import List

import yfinance as yf
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

###############################################################################
#                              ENV & LOGGING                                  #
###############################################################################
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # DEBUG level for detailed messages

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)

# Create formatter and add it to handlers
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)

###############################################################################
#                        QUERY CONSTRUCTION FUNCTIONS                         #
###############################################################################
def build_small_cap_query(lower_bound: float, upper_bound: float) -> dict:
    """
    Constructs a query for US tech stocks with market cap between lower_bound and upper_bound.
    """
    try:
        sector_eq = {'operator': 'eq', 'operands': ['sector', 'Technology']}
        region_eq = {'operator': 'eq', 'operands': ['region', 'us']}
        market_cap_gt = {'operator': 'gt', 'operands': ['intradaymarketcap', lower_bound]}
        market_cap_lt = {'operator': 'lt', 'operands': ['intradaymarketcap', upper_bound]}
        avg_vol_gt = {'operator': 'gt', 'operands': ['avgdailyvol3m', 100000]}
        query = {
            'operator': 'and',
            'operands': [sector_eq, region_eq, market_cap_gt, market_cap_lt, avg_vol_gt]
        }
        logger.debug(f"Built small cap query for range ${lower_bound} - ${upper_bound}")
        return query
    except Exception as e:
        logger.error(f"Error constructing small cap query (${lower_bound}-${upper_bound}): {e}")
        sys.exit(1)

def build_low_volatility_query(lower_bound: float, upper_bound: float, volatility_threshold: float = 20) -> dict:
    """
    Constructs a query similar to build_small_cap_query with an added volatility filter.
    """
    try:
        base_query = build_small_cap_query(lower_bound, upper_bound)
        low_vol = {'operator': 'lt', 'operands': ['historicalvolatility', volatility_threshold]}
        base_query['operands'].append(low_vol)
        logger.debug(f"Built low volatility query with threshold {volatility_threshold} for range ${lower_bound}-${upper_bound}")
        return base_query
    except Exception as e:
        logger.error(f"Error constructing low volatility query: {e}")
        sys.exit(1)

def build_high_volume_query(lower_bound: float, upper_bound: float, volume_threshold: float = 20000) -> dict:
    """
    Constructs a query variant that requires a higher average volume.
    """
    try:
        base_query = build_small_cap_query(lower_bound, upper_bound)
        high_vol = {'operator': 'gt', 'operands': ['avgdailyvol3m', volume_threshold]}
        # Replace the default avg_vol condition (100,000) with the higher threshold.
        operands = [op for op in base_query['operands'] if not (op.get('operands') and op['operands'][1] == 100000)]
        operands.append(high_vol)
        base_query['operands'] = operands
        logger.debug(f"Built high volume query with threshold {volume_threshold} for range ${lower_bound}-${upper_bound}")
        return base_query
    except Exception as e:
        logger.error(f"Error constructing high volume query: {e}")
        sys.exit(1)

###############################################################################
#                         TICKER FETCHING LOGIC                             #
###############################################################################
def fetch_tickers_for_query(query: dict, fetch_size: int = 249) -> List[str]:
    """
    Fetches tickers using the given query.
    """
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
            logger.warning("No stocks found for the given query criteria.")
            return []
        tickers = [quote['symbol'] for quote in response['quotes']]
        logger.debug(f"Fetched tickers: {tickers}")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching tickers with query: {e}")
        return []

def has_historical_data(ticker: str, start_date: str = "2019-01-01") -> bool:
    """
    Checks whether the data downloaded for the ticker is not empty.
    """
    try:
        logger.debug(f"Downloading historical data for {ticker} from {start_date}...")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty:
            logger.debug(f"No historical data found for {ticker}.")
            return False
        # Additionally, check that we have a minimum number of data points (e.g. 250).
        if len(df) < 1000:
            logger.debug(f"Ticker {ticker} has insufficient data points ({len(df)}) since {start_date}.")
            return False
        logger.debug(f"Historical data for {ticker} downloaded successfully with {len(df)} data points.")
        return True
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        return False

###############################################################################
#                         BUILD STOCK LIBRARY                               #
###############################################################################
def build_stock_library(
    desired_count: int = 700,
    fetch_size: int = 249,
    market_cap_threshold: float = 2e9,
    market_cap_minimum: float = 50000000
) -> List[str]:
    """
    Builds a library of US Small Cap Tech stocks meeting the criteria.
    
    The library is built by iterating over these queries:
      1. Lower small cap (50M to 500M)
      2. Mid small cap (500M to 1B)
      3. Upper small cap (1B to 2B)
      4. Low volatility variant over (50M to 2B)
      5. High volume variant over (50M to 2B, volume > 200K)
    """
    library = set()  # Using a set for uniqueness
    queries = [
        build_small_cap_query(50000000, 500000000),     # lower small cap
        build_small_cap_query(500000000, 750000000),    # mid small cap
        build_small_cap_query(750000000, 1000000000),
        build_small_cap_query(1000000000, 1500000000),   # upper small cap
        build_small_cap_query(1500000000, 2000000000),
        build_high_volume_query(50000000, 1000000000),
        build_high_volume_query(1000000000, 2000000000),
        build_low_volatility_query(50000000, 1000000000),
        build_low_volatility_query(1000000000, 2000000000)
    ]
    
    # Non-US suffixes to filter out tickers.
    non_us_suffixes = [
        ".AR", ".AT", ".AU", ".BE", ".BR", ".CA", ".CH", ".CL", ".CN", ".CZ",
        ".DE", ".DK", ".EE", ".EG", ".ES", ".FI", ".FR", ".GB", ".GR", ".HK",
        ".HU", ".ID", ".IE", ".IL", ".IN", ".IS", ".IT", ".JP", ".KR", ".KW",
        ".LK", ".LT", ".LV", ".MX", ".MY", ".NL", ".NO", ".NZ", ".PE", ".PH",
        ".PK", ".PL", ".PT", ".QA", ".RO", ".RU", ".SA", ".SE", ".SG", ".SR",
        ".TH", ".TR", ".TW", ".VE", ".VN", ".ZA", ".OL", ".PA"
    ]
    
    for idx, query in enumerate(queries, 1):
        logger.info(f"Processing query variant {idx}/{len(queries)}...")
        tickers = fetch_tickers_for_query(query, fetch_size=fetch_size)
        logger.info(f"Fetched {len(tickers)} tickers for query variant {idx}.")
        
        for sym in tickers:
            # Reject tickers with foreign suffixes.
            if any(sym.upper().endswith(suf) for suf in non_us_suffixes):
                logger.debug(f"Ticker {sym} rejected due to foreign suffix.")
                continue
            
            try:
                ticker_obj = yf.Ticker(sym)
                info = ticker_obj.info
                market_cap = info.get("marketCap", 0)
                if not market_cap:
                    logger.debug(f"Ticker {sym} rejected: market cap information missing.")
                    continue
                if not (market_cap_minimum <= market_cap < market_cap_threshold):
                    logger.debug(f"Ticker {sym} rejected: market cap ${market_cap} out of range.")
                    continue

                if not has_historical_data(sym, start_date="2019-01-01"):
                    logger.debug(f"Ticker {sym} rejected: inadequate historical data.")
                    continue

                df = yf.download(sym, start="2019-01-01", progress=False)
                if df.empty:
                    logger.debug(f"Ticker {sym} rejected: historical data download returned empty DataFrame.")
                    continue
                # Ensure the 'Close' column is a Series by squeezing.
                close_series = df['Close'].squeeze()
                current_price = float(close_series.iloc[-1])
                if current_price < 3.0:
                    logger.debug(f"Ticker {sym} rejected: price ${current_price:.2f} is below $5.")
                    continue

                library.add(sym)
                logger.info(f"Ticker {sym} accepted: Market Cap=${market_cap}, Price=${current_price:.2f}.")
                if len(library) >= desired_count:
                    logger.info("Desired ticker count reached.")
                    break
            except Exception as ee:
                logger.warning(f"Error processing ticker {sym}: {ee}")
                continue
        
        if len(library) >= desired_count:
            break
    
    if len(library) < desired_count:
        logger.warning(f"Only found {len(library)} tickers meeting criteria (desired {desired_count}).")
    else:
        logger.info(f"Successfully built library with {len(library)} tickers.")
    
    return list(library)

###############################################################################
#                                MAIN                                        #
###############################################################################
def main():
    logger.info("=== Building Stock Library for US Small Cap Tech Stocks ===")
    
    desired_count = 700
    fetch_size = 249  # number of tickers to try fetching per query variant
    
    stock_library = build_stock_library(desired_count=desired_count, fetch_size=fetch_size)
    
    if not stock_library:
        logger.info("No valid tickers found. Exiting.")
        return
    
    library_df = pd.DataFrame({'Ticker': stock_library})
    output_file = "stock_library.csv"
    try:
        library_df.to_csv(output_file, index=False)
        logger.info(f"Stock library saved to '{output_file}'. Total tickers: {len(stock_library)}")
    except Exception as e:
        logger.error(f"Error saving stock library to CSV: {e}")

if __name__ == "__main__":
    main()
