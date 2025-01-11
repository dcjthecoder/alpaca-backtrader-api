"""
test_alpaca.py

Objective:
    1) Connect to the Alpaca API using credentials from env.txt.
    2) Select two small-cap tech stocks.
    3) Download tech sector data using yfinance.
    4) Check Alpaca account balance.
    5) Place a buy order for one of the selected tech stocks.
    6) Connect to Alpaca's WebSocket stream to receive real-time updates.
    7) Log all actions for debugging purposes.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
import json
import time

import yfinance as yf
import alpaca_trade_api as tradeapi
import pandas as pd  # Added import for pandas
import websocket  # Import websocket-client library

###############################################################################
#                              ENV & LOGGING                                  #
###############################################################################
LOG_FILE = "test_alpaca.log"

# Configure RotatingFileHandler to prevent log files from growing indefinitely
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("TestAlpacaLogger")
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

###############################################################################
#                           LOAD ENVIRONMENT VARIABLES                       #
###############################################################################
def load_api_credentials(env_file: str = "env.txt"):
    """
    Load Alpaca API credentials from the specified environment file.
    """
    if not os.path.exists(env_file):
        logger.error(f"Environment file '{env_file}' not found. Exiting.")
        sys.exit(1)
    
    load_dotenv(env_file)
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    
    if not api_key or not api_secret:
        logger.error("Alpaca API keys are missing in env.txt. Exiting.")
        sys.exit(1)
    
    return api_key, api_secret

###############################################################################
#                           SELECT SMALL-CAP TECH STOCKS                       #
###############################################################################
def select_small_cap_tech_stocks() -> list:
    """
    Select two small-cap tech stocks.
    For demonstration purposes, we'll select two predefined tickers.
    """
    # Predefined list of small-cap tech tickers
    # Ensure these are indeed small-cap and in the tech sector
    small_cap_tech_tickers = ["BTCM", "KOSS"]  # Example tickers
    
    # Validate the selected tickers
    valid_tickers = []
    for ticker in small_cap_tech_tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "").lower()
            market_cap = info.get("marketCap", 0)
            
            if sector != "technology":
                logger.warning(f"Ticker {ticker} is not in the Technology sector. Skipping.")
                continue
            if market_cap >= 2e9:  # 2 billion USD threshold for small-cap
                logger.warning(f"Ticker {ticker} has Market Cap=${market_cap} which exceeds small-cap threshold. Skipping.")
                continue
            
            valid_tickers.append(ticker)
            logger.info(f"Selected Ticker: {ticker}, Sector: {sector.capitalize()}, Market Cap=${market_cap}")
        except Exception as e:
            logger.error(f"Error fetching info for ticker {ticker}: {e}")
    
    if len(valid_tickers) < 2:
        logger.error("Not enough valid small-cap tech tickers selected. Exiting.")
        sys.exit(1)
    
    return valid_tickers[:2]

###############################################################################
#                           DOWNLOAD TECH SECTOR DATA                        #
###############################################################################
def download_tech_sector_data(etf_ticker: str = "XLK", period: str = "1mo"):
    """
    Download tech sector data using yfinance.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 1 month data
        
        logger.info(f"Downloading data for Tech Sector ETF '{etf_ticker}' from {start_date.date()} to {end_date.date()}")
        sector_data = yf.download(
            etf_ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False
        )
        
        if sector_data.empty:
            logger.warning(f"No data fetched for ETF {etf_ticker}.")
        else:
            logger.info(f"Downloaded {len(sector_data)} days of data for {etf_ticker}.")
            logger.debug(sector_data.head())
        
        return sector_data
    except Exception as e:
        logger.error(f"Error downloading sector data for {etf_ticker}: {e}")
        return pd.DataFrame()

###############################################################################
#                           CHECK ACCOUNT BALANCE                             #
###############################################################################
def check_account_balance(api: tradeapi.REST):
    """
    Check and log the Alpaca account balance.
    """
    try:
        account = api.get_account()
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Cash Balance: ${account.cash}")
        logger.info(f"Portfolio Value: ${account.portfolio_value}")
        return account
    except Exception as e:
        logger.error(f"Error fetching account information: {e}")
        sys.exit(1)

###############################################################################
#                           PLACE BUY ORDER                                   #
###############################################################################
def place_buy_order(api: tradeapi.REST, ticker: str, qty: int = 1):
    """
    Place a buy order for the specified ticker and quantity.
    """
    try:
        logger.info(f"Placing BUY order for {qty} shares of {ticker}")
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        logger.info(f"Buy order submitted: {order}")
        return order
    except Exception as e:
        logger.error(f"Error placing buy order for {ticker}: {e}")
        return None

###############################################################################
#                           WEBSOCKET STREAM                                  #
###############################################################################
def on_message(ws, message):
    """
    Callback executed when a message is received from the WebSocket.
    """
    try:
        data = json.loads(message)
        logger.info(f"WebSocket Message Received: {data}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON message: {message}")

def on_error(ws, error):
    """
    Callback executed when an error occurs with the WebSocket.
    """
    logger.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    """
    Callback executed when the WebSocket connection is closed.
    """
    logger.info(f"WebSocket Closed with code: {close_status_code}, message: {close_msg}")
    # Optional: Implement reconnection logic here

def on_open(ws, api_key, api_secret, stream_type):
    """
    Callback executed when the WebSocket connection is opened.
    Handles authentication and subscription.
    """
    logger.info("WebSocket Connection Opened. Authenticating...")
    auth_message = {
        "action": "auth",
        "key": api_key,
        "secret": api_secret
    }
    ws.send(json.dumps(auth_message))
    
    # Wait a moment before subscribing
    time.sleep(1)
    
    # Define subscription message based on stream type
    if stream_type == "test":
        # Subscribe to test symbol
        subscribe_message = {
            "action": "subscribe",
            "trades": ["FAKEPACA"],
            "quotes": ["FAKEPACA"],
            "bars": ["FAKEPACA"]
        }
    elif stream_type == "live":
        # Subscribe to real channels and symbols
        subscribe_message = {
            "action": "subscribe",
            "trades": ["ALL"],
            "quotes": ["ALL"],
            "bars": ["ALL"],
            "orders": ["ALL"],  # Only valid in live stream
            "positions": ["ALL"]
        }
    else:
        logger.error(f"Unknown stream type: {stream_type}")
        ws.close()
        return
    
    ws.send(json.dumps(subscribe_message))
    logger.info("Subscribed to channels.")

def start_websocket(api_key, api_secret, stream_type="test"):
    """
    Initializes and runs the WebSocket connection in a separate thread.
    
    Parameters:
        api_key (str): Your Alpaca API Key ID.
        api_secret (str): Your Alpaca API Secret Key.
        stream_type (str): "test" for Test Stream or "live" for Live Stream.
    """
    if stream_type == "test":
        websocket_url = "wss://stream.data.alpaca.markets/v2/test"
    elif stream_type == "live":
        websocket_url = "wss://stream.data.alpaca.markets/v2/iex"
    else:
        logger.error(f"Invalid stream type: {stream_type}. Choose 'test' or 'live'.")
        sys.exit(1)
    
    ws = websocket.WebSocketApp(
        websocket_url,
        on_open=lambda ws: on_open(ws, api_key, api_secret, stream_type),
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run WebSocket in a separate thread to prevent blocking
    wst = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 60, "ping_timeout": 10})
    wst.daemon = True
    wst.start()
    logger.info("WebSocket thread started.")

###############################################################################
#                                 MAIN                                        #
###############################################################################
def main():
    logger.info("=== Starting Alpaca Test Script ===")
    
    # 1. Load API Credentials
    api_key, api_secret = load_api_credentials()
    logger.info("Alpaca API credentials loaded successfully.")
    
    # 2. Initialize Alpaca API Client (Paper Trading)
    base_url = "https://paper-api.alpaca.markets"  # Use paper trading URL
    try:
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        api.get_account()  # Simple call to verify connection
        logger.info("Connected to Alpaca API (Paper Trading).")
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {e}")
        sys.exit(1)
    
    # 3. Select Two Small-Cap Tech Stocks
    selected_tickers = select_small_cap_tech_stocks()
    logger.info(f"Selected Tickers for Trading: {selected_tickers}")
    
    # 4. Download Tech Sector Data
    sector_data = download_tech_sector_data()
    if not sector_data.empty:
        # Example: Calculate and log the sector's performance over the period
        try:
            # Ensure 'Close' column exists
            if 'Close' not in sector_data.columns:
                logger.error(f"'Close' column not found in sector data for XLK.")
                sector_performance = 0.0
            else:
                # Extract scalar values to avoid FutureWarnings
                start_price = sector_data['Close'].iloc[0]
                end_price = sector_data['Close'].iloc[-1]
                
                # Ensure the extracted values are floats
                if isinstance(start_price, pd.Series):
                    start_price = float(start_price.iloc[0])
                else:
                    start_price = float(start_price)
                
                if isinstance(end_price, pd.Series):
                    end_price = float(end_price.iloc[0])
                else:
                    end_price = float(end_price)
                
                performance = ((end_price - start_price) / start_price) * 100
                logger.info(f"Tech Sector Performance over the period: {performance:.2f}%")
        except Exception as e:
            logger.error(f"Error calculating sector performance: {e}")
    else:
        logger.warning("No sector data to analyze.")
    
    # 5. Check Account Balance
    account = check_account_balance(api)
    
    # 6. Place a Buy Order for the First Selected Ticker
    # For testing, we'll buy a small number of shares to minimize risk
    ticker_to_buy = selected_tickers[0]
    quantity_to_buy = 1  # Adjust as needed
    
    # Function to fetch last trade price using yfinance
    def get_last_trade_price(ticker):
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period='1d')
            if data.empty:
                logger.error(f"No trade data found for {ticker}")
                return None
            last_price = data['Close'].iloc[-1]
            return last_price
        except Exception as e:
            logger.error(f"Error fetching last trade price for {ticker}: {e}")
            return None
    
    # Check if there's enough cash to place the order
    try:
        # Attempt to fetch last trade via Alpaca
        try:
            last_trade = api.get_last_trade(ticker_to_buy)
            current_price = float(last_trade.price)
            source = "last_trade"
        except AttributeError:
            # Fallback to yfinance if get_last_trade is unavailable
            current_price = get_last_trade_price(ticker_to_buy)
            source = "yfinance"
            if current_price is None:
                logger.error(f"Could not retrieve price for {ticker_to_buy}. Exiting.")
                sys.exit(1)
        
        total_cost = current_price * quantity_to_buy
        available_cash = float(account.cash)
        
        logger.info(f"Current Price for {ticker_to_buy} ({source}): ${current_price:.2f}")
        logger.info(f"Total Cost for {quantity_to_buy} shares: ${total_cost:.2f}")
        logger.info(f"Available Cash: ${available_cash:.2f}")
        
        if available_cash < total_cost:
            logger.error(f"Insufficient funds to buy {quantity_to_buy} shares of {ticker_to_buy} at ${current_price:.2f} each.")
            sys.exit(1)
        else:
            order = place_buy_order(api, ticker_to_buy, qty=quantity_to_buy)
    except Exception as e:
        logger.error(f"Error checking price or placing order for {ticker_to_buy}: {e}")
    
    # 7. Start WebSocket Stream to Receive Real-Time Updates
    if order:
        # Choose stream type: "test" or "live"
        # For Test Stream, use "test" and subscribe to FAKEPACA
        # For Live Stream, use "live" and subscribe to relevant channels
        stream_type = "live"  # Change to "test" if using Test Stream
        
        start_websocket(api_key, api_secret, stream_type)
    
    # Keep the main thread alive to continue receiving WebSocket messages
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
        sys.exit(0)
    
    logger.info("=== Alpaca Test Script Completed ===")

if __name__ == "__main__":
    main()
