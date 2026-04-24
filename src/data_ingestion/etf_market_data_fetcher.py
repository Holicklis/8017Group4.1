import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def normalize_symbol(symbol: str) -> str:
    """Normalize HKEX symbol to 4-digit code used in folder names."""
    return str(symbol).strip().zfill(4)


def to_yahoo_ticker(symbol: str) -> str:
    """Convert HKEX symbol to Yahoo Finance ticker format."""
    return f"{normalize_symbol(symbol)}.HK"


def get_etf_root() -> Path:
    """Get ETF data root, supporting both etf/ and ETF/ directories."""
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    return data_root / "etf" if (data_root / "etf").exists() else data_root / "ETF"


def load_symbols_from_csv(csv_path: Path, column_name: str = "instruments") -> list[str]:
    """Load ticker symbols from a CSV file."""
    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise ValueError(f"CSV must contain '{column_name}' column. Found: {list(df.columns)}")
    return [normalize_symbol(value) for value in df[column_name].dropna().astype(str).tolist()]


def save_hkex_data(symbol: str, start_date: str, end_date: str) -> None:
    formatted_symbol = normalize_symbol(symbol)
    ticker_symbol = to_yahoo_ticker(formatted_symbol)

    etf_root = get_etf_root()

    ohlcv_path = etf_root / "ohlcv" / formatted_symbol
    mkt_cap_path = etf_root / "market_cap" / formatted_symbol
    ohlcv_path.mkdir(parents=True, exist_ok=True)
    mkt_cap_path.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data for: {ticker_symbol}")

    df_ohlcv = yf.download(ticker_symbol, start=start_date, end=end_date)
    df_ohlcv = df_ohlcv.droplevel(1, axis=1).reset_index()
    if not df_ohlcv.empty:
        df_ohlcv.to_parquet(str(ohlcv_path / "ohlcv.parquet"))
        print(f"Saved OHLCV data to {ohlcv_path}")
    else:
        print(f"No OHLCV data found for {ticker_symbol}")

    ticker_obj = yf.Ticker(ticker_symbol)
    info = ticker_obj.info
    mkt_cap_value = info.get("marketCap")

    if mkt_cap_value:
        df_mkt_cap = pd.DataFrame([{
            "symbol": ticker_symbol,
            "market_cap": mkt_cap_value,
            "currency": info.get("currency", "HKD"),
            "timestamp": pd.Timestamp.now(),
        }])

        df_mkt_cap.to_parquet(str(mkt_cap_path / "market_cap.parquet"))
        print(f"Saved Market Cap data to {mkt_cap_path}")
    else:
        print(f"Market Cap not available for {ticker_symbol}")


def save_hkex_data_batch(symbols: Iterable[str], start_date: str, end_date: str) -> None:
    """Fetch and save Yahoo data for a list of HKEX symbols."""
    for symbol in symbols:
        try:
            save_hkex_data(symbol=symbol, start_date=start_date, end_date=end_date)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed for {symbol}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and save HKEX market data to Parquet.")
    parser.add_argument(
        "--ticker",
        type=str,
        help="A single HKEX ticker symbol (e.g., 2800 or 0005).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "etf" / "instruments" / "all_hk_etf.csv",
        help="Path to CSV containing a ticker column named 'instruments'.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date for OHLCV download (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-04-19",
        help="End date for OHLCV download (YYYY-MM-DD).",
    )

    args = parser.parse_args()

    if args.ticker:
        save_hkex_data(symbol=args.ticker, start_date=args.start_date, end_date=args.end_date)
    else:
        symbols = load_symbols_from_csv(args.csv)
        save_hkex_data_batch(symbols=symbols, start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
