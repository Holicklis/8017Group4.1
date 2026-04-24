from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFDataException

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def to_yahoo_ticker(folder_ticker: str) -> str:
    """Convert folder ticker to Yahoo format when needed."""
    ticker = folder_ticker.strip()
    if ticker.isdigit():
        return f"{ticker}.HK"
    return ticker


def fetch_top_holdings(yahoo_ticker: str, folder_ticker: str) -> pd.DataFrame:
    """
    Fetch ETF top holdings from yfinance funds_data.top_holdings.
    Returns an empty DataFrame when holdings are unavailable.
    """
    LOGGER.info("Fetching top holdings for %s", yahoo_ticker)
    try:
        etf = yf.Ticker(yahoo_ticker)
        holdings_df = etf.funds_data.top_holdings
    except YFDataException as exc:
        LOGGER.warning("Skipping %s: %s", yahoo_ticker, exc)
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Skipping %s due to unexpected error: %s", yahoo_ticker, exc)
        return pd.DataFrame()

    if holdings_df is None or holdings_df.empty:
        LOGGER.warning("No holdings found for %s", yahoo_ticker)
        return pd.DataFrame()

    df = holdings_df.reset_index().copy()
    df.insert(0, "etf_ticker", folder_ticker)
    df.insert(1, "yahoo_ticker", yahoo_ticker)
    return df


def resolve_default_output_path(ticker_symbol: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "etf" / "holdings" / "top10" / ticker_symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "top_holdings.parquet"


def load_symbols_from_csv(csv_path: Path, column_name: str = "instruments") -> list[str]:
    """Load ticker symbols from instruments CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Instruments CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise ValueError(f"CSV must contain '{column_name}' column. Found: {list(df.columns)}")
    return sorted(df[column_name].dropna().astype(str).str.strip().tolist())


def save_ticker_holdings(ticker_symbol: str, output_root: Path | None = None) -> None:
    yahoo_ticker = to_yahoo_ticker(ticker_symbol)
    holdings_df = fetch_top_holdings(yahoo_ticker=yahoo_ticker, folder_ticker=ticker_symbol)
    if holdings_df.empty:
        LOGGER.error("No holdings data to save for %s", ticker_symbol)
        return

    output_path = (
        (output_root / ticker_symbol / "top_holdings.parquet")
        if output_root is not None
        else resolve_default_output_path(ticker_symbol)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    holdings_df.to_parquet(output_path, index=False)
    LOGGER.info("Saved %d rows to %s", len(holdings_df), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ETF top holdings and save to parquet.")
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Single ETF ticker symbol (e.g. 2800). If omitted, read all symbols from --csv.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "etf" / "instruments" / "all_hk_etf.csv",
        help="Path to instruments CSV (default: data/etf/instruments/all_hk_etf.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root folder. Data is saved as <output-root>/<ticker>/top_holdings.parquet.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.ticker:
        save_ticker_holdings(ticker_symbol=args.ticker, output_root=args.output_root)
        return

    tickers = load_symbols_from_csv(args.csv)
    if not tickers:
        LOGGER.warning("No ticker symbols found in %s", args.csv)
        return

    LOGGER.info("Found %d ticker symbols in %s", len(tickers), args.csv)
    for ticker in tickers:
        save_ticker_holdings(ticker_symbol=ticker, output_root=args.output_root)


if __name__ == "__main__":
    main()
