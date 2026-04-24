import yfinance as yf
import pandas as pd
import os
import argparse
from pathlib import Path

def save_hkex_data(symbol):
    # Ensure symbol is 4 or 5 digits (e.g., 2800 -> 02800)
    # yfinance usually expects the 5-digit format for .HK
    formatted_symbol = symbol.zfill(4) if len(symbol) <= 4 else symbol
    ticker_symbol = f"{formatted_symbol}.HK"
    
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    etf_root = data_root / "etf" if (data_root / "etf").exists() else data_root / "ETF"

    # Define Paths
    ohlcv_path = etf_root / "ohlcv" / formatted_symbol
    mkt_cap_path = etf_root / "market_cap" / formatted_symbol
    
    # Create directories if they don't exist
    os.makedirs(str(ohlcv_path), exist_ok=True)
    os.makedirs(str(mkt_cap_path), exist_ok=True)

    print(f"Fetching data for: {ticker_symbol}")
    
    # 1. Download OHLCV Data
    df_ohlcv = yf.download(ticker_symbol, start="2020-01-01", end="2026-04-19")
    
    if not df_ohlcv.empty:
        df_ohlcv.to_parquet(str(ohlcv_path / "ohlcv.parquet"))
        print(f"✅ Saved OHLCV data to {ohlcv_path}")
    else:
        print(f"❌ No OHLCV data found for {ticker_symbol}")

    # 2. Fetch and Save Market Cap
    ticker_obj = yf.Ticker(ticker_symbol)
    info = ticker_obj.info
    mkt_cap_value = info.get('marketCap')
    
    if mkt_cap_value:
        df_mkt_cap = pd.DataFrame([{
            'symbol': ticker_symbol,
            'market_cap': mkt_cap_value,
            'currency': info.get('currency', 'HKD'),
            'timestamp': pd.Timestamp.now()
        }])
        
        df_mkt_cap.to_parquet(str(mkt_cap_path / "market_cap.parquet"))
        print(f"✅ Saved Market Cap data to {mkt_cap_path}")
    else:
        print(f"⚠️ Market Cap not available for {ticker_symbol}")

if __name__ == "__main__":
    # Set up CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Fetch and save HKEX data to Parquet.")
    parser.add_argument(
        "ticker", 
        type=str, 
        help="The HKEX ticker symbol (e.g., 2800 or 0005)"
    )
    
    args = parser.parse_args()
    
    # Run the function
    save_hkex_data(args.ticker)