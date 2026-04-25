"""
Unified ETF data pipeline entrypoint.

This script orchestrates:
1) HKEX ETP metadata export (xlsx)
2) HKEX ETF PDF document scraping
3) PDF text extraction to CSV
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_ingestion.provider.hkex.hkex_etf.etf_document_scraper import (
    configure_logging,
    scrape_many_tickers,
)
from data_ingestion.provider.hkex.hkex_etf.etf_metadata_export import download_full_etp_list
from text_extraction.pdf_text_extractor import ETFPDFProcessor

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_summary_file() -> Path:
    data_dir = _project_root() / "data"
    lower_path = data_dir / "etf" / "summary" / "ETP_Data_Export.xlsx"
    legacy_path = data_dir / "ETF" / "Summary" / "ETP_Data_Export.xlsx"
    if lower_path.exists():
        return lower_path
    if legacy_path.exists():
        return legacy_path
    return lower_path


def _extract_ticker_list(summary_file: Path) -> List[str]:
    """
    Try common ticker columns from HKEX export and return normalized ticker list.
    """
    df_summary = pd.read_excel(summary_file, skipfooter=2)
    ticker_candidates = [
        "Stock code*",
        "Stock Code",
        "StockCode",
        "stock_code",
        "Code",
        "code",
    ]

    for column_name in ticker_candidates:
        if column_name in df_summary.columns:
            df_filtered = df_summary.copy()
            if "Stock code*" in df_filtered.columns and "Base currency*" in df_filtered.columns:
                df_filtered = df_filtered.query("`Stock code*` < 8000 and `Base currency*` == 'HKD'")

            series_codes = (
                pd.to_numeric(df_filtered[column_name], errors="coerce")
                .dropna()
                .astype(int)
                .drop_duplicates()
                .sort_values()
            )
            return [str(value).zfill(5) for value in series_codes.tolist()]

    raise ValueError(
        f"Cannot find ticker column in {summary_file}. "
        f"Expected one of: {', '.join(ticker_candidates)}"
    )


def run_pipeline(
    summary_file: Optional[Path],
    skip_export: bool,
    skip_documents: bool,
    skip_text_extract: bool,
    skip_profile_generation: bool,
    headless: bool,
) -> None:
    """Run the full ETF processing pipeline with optional stage skipping."""
    resolved_summary_file = summary_file or _default_summary_file()
    resolved_summary_file.parent.mkdir(parents=True, exist_ok=True)

    if not skip_export:
        logger.info("Stage 1/3: Downloading HKEX ETP metadata export")
        downloaded_file = download_full_etp_list(output_dir=resolved_summary_file.parent, headless=headless)
        if downloaded_file:
            resolved_summary_file = downloaded_file

    if not skip_documents:
        logger.info("Stage 2/3: Downloading ETF PDF documents")
        ticker_list = _extract_ticker_list(resolved_summary_file)
        scrape_many_tickers(ticker_list, headless=headless)

    if not skip_text_extract:
        logger.info("Stage 3/3: Extracting PDF text into CSV and ETF profiles")
        processor = ETFPDFProcessor(data_root=_project_root() / "data")
        processor.run_pipeline(generate_profiles=not skip_profile_generation)

    logger.info("ETF data pipeline completed successfully")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run full ETF data pipeline")
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Path to ETP_Data_Export.xlsx (default: data/etf/summary/ETP_Data_Export.xlsx)",
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip stage 1 (ETP export)")
    parser.add_argument("--skip-documents", action="store_true", help="Skip stage 2 (PDF scraping)")
    parser.add_argument("--skip-text-extract", action="store_true", help="Skip stage 3 (PDF to CSV)")
    parser.add_argument(
        "--skip-profile-generation",
        action="store_true",
        help="Skip ETF profile generation during text extraction stage",
    )
    parser.add_argument("--no-headless", action="store_true", help="Run Chrome with UI")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    arguments = parse_args()
    run_pipeline(
        summary_file=arguments.summary_file,
        skip_export=arguments.skip_export,
        skip_documents=arguments.skip_documents,
        skip_text_extract=arguments.skip_text_extract,
        skip_profile_generation=arguments.skip_profile_generation,
        headless=not arguments.no_headless,
    )
