#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A short script to test yfinance's Search class and fetch historical news.
For each ticker, we search using queries that include the ticker and a March 26 date.
"""

import yfinance as yf

# Choose three tickers
tickers = ["AAPL", "GOOGL", "MSFT"]
# List of dates (March 26 of the selected years)
years = [2020, 2021, 2022, 2023, 2024]
march_26_dates = [f"March 26, {year}" for year in years]

# Loop through each ticker and date, build a query and print out the news results.
for ticker in tickers:
    print(f"\n=== News for {ticker} ===")
    for date_str in march_26_dates:
        # Build the query string. You can customize this as needed.
        query = f"{ticker}. {date_str}"
        print(f"\n-- Query: {query} --")
        try:
            # Create an instance of yfinance.Search with news_count (default is 8)
            search_result = yf.Search(query=query, news_count=8, enable_fuzzy_query=True)
            
            # Access the news from the search result.
            # The news attribute should be a list of dictionaries (one per article).
            news_items = search_result.news
            
            if news_items:
                for idx, article in enumerate(news_items, start=1):
                    title = article.get("title", "No title")
                    publisher = article.get("publisher", "Unknown")
                    link = article.get("link", "No link")
                    print(f"{idx}. {title}")
                    print(f"   Publisher: {publisher}")
                    print(f"   Link: {link}")
            else:
                print("No news found for this query.")
        except Exception as e:
            print(f"An error occurred while searching news for query '{query}': {e}")

print("\n=== End of Search ===")
