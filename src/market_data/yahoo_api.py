import yfinance as yf
import pandas as pd
import os

def save_hkex_data(symbol):
    # 1. Format Ticker (e.g., 2800 -> 02800.HK)
    ticker_symbol = symbol + ".HK"
    
    # Define Paths
    ohlcv_path = f"data/ETF/OHLCV/{symbol}"
    mkt_cap_path = f"data/ETF/MKT_CAP/{symbol}"
    
    # Create directories if they don't exist
    os.makedirs(ohlcv_path, exist_ok=True)
    os.makedirs(mkt_cap_path, exist_ok=True)


    # 2. Download OHLCV Data
    # Note: yf.download is great for batches, but for a single ticker with metadata, 
    print(ticker_symbol)
    df_ohlcv = yf.download(ticker_symbol, start="2025-01-01", end="2026-04-19")
    print(df_ohlcv)
    
    if not df_ohlcv.empty:
        # Save OHLCV to Parquet
        df_ohlcv.to_parquet(os.path.join(ohlcv_path, "OHLCV.parquet"))
        print(f"Saved OHLCV data to {ohlcv_path}")
    else:
        print(f"No OHLCV data found for {ticker_symbol}")

    # 3. Fetch and Save Market Cap
    info = yf.Ticker(ticker_symbol).info
    mkt_cap_value = info.get('marketCap')
    
    if mkt_cap_value:
        # Create a small DataFrame for the Market Cap
        df_mkt_cap = pd.DataFrame([{
            'symbol': ticker_symbol,
            'market_cap': mkt_cap_value,
            'currency': info.get('currency', 'HKD'),
            'timestamp': pd.Timestamp.now()
        }])
        
        # Save Mkt Cap to Parquet
        df_mkt_cap.to_parquet(os.path.join(mkt_cap_path, "market_cap.parquet"))
        print(f"Saved Market Cap data to {mkt_cap_path}")
    else:
        print(f"Market Cap not available for {ticker_symbol}")

# Example Usage
save_hkex_data("2800")