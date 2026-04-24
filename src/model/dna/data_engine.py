"""
Financial DNA feature engineering for HKEX ETFs.

Module 1 of the 3-module system:
- Load HKEX metadata + per-ticker OHLCV parquet
- Engineer core numeric DNA features
- Keep context labels for downstream cluster interpretation
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
REQUIRED_METADATA_COLUMNS = [
    "Stock code*",
    "Stock short name*",
    "Product sub-category*",
    "Dividend yield (%)*",
    "Ongoing Charges Figures (%)*",
    "AUM",
    "Closing price",
    "Premium/discount %",
    "Asset class*",
    "Geographic focus*",
    "Investment focus*",
    "Management Style",
    "Thematic",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_etf_data_root() -> Path:
    data_root = _project_root() / "data"
    return data_root / "etf" if (data_root / "etf").exists() else data_root / "ETF"


def _to_hkex_code(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    return digits.zfill(4)


def _to_yahoo_ticker(value: object) -> Optional[str]:
    code = _to_hkex_code(value)
    if code is None:
        return None
    return f"{code}.HK"


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


class ETFDataProcessor:
    """Build Financial DNA features from metadata and OHLCV parquet files."""

    def __init__(
        self,
        metadata_path: Optional[Path] = None,
        ohlcv_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        instruments_path: Optional[Path] = None,
        rolling_window: int = 30,
    ) -> None:
        etf_root = _default_etf_data_root()
        self.metadata_path = metadata_path or etf_root / "summary" / "ETP_Data_Export.xlsx"
        self.ohlcv_dir = ohlcv_dir or etf_root / "ohlcv"
        self.output_path = output_path or etf_root / "processed" / "financial_dna.parquet"
        self.instruments_path = instruments_path or etf_root / "instruments" / "all_hk_etf.csv"
        self.rolling_window = rolling_window

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> None:
        missing = [name for name in REQUIRED_METADATA_COLUMNS if name not in df.columns]
        if missing:
            raise ValueError(f"Metadata file is missing required columns: {missing}")

    def _load_metadata(self) -> pd.DataFrame:
        logger.info("Loading metadata: %s", self.metadata_path)
        df = pd.read_excel(self.metadata_path, skipfooter=2)
        self._validate_required_columns(df)

        out = df.loc[:, REQUIRED_METADATA_COLUMNS].copy()
        out = out.rename(
            columns={
                "Stock code*": "stock_code_raw",
                "Stock short name*": "stock_short_name",
                "Product sub-category*": "product_sub_category",
                "Dividend yield (%)*": "dividend_yield_pct",
                "Ongoing Charges Figures (%)*": "ongoing_charges_pct",
                "AUM": "aum",
                "Closing price": "closing_price_metadata",
                "Premium/discount %": "premium_discount_pct",
                "Asset class*": "asset_class",
                "Geographic focus*": "geographic_focus",
                "Investment focus*": "investment_focus",
                "Management Style": "management_style",
                "Thematic": "thematic",
            }
        )

        out["stock_code_raw"] = pd.to_numeric(out["stock_code_raw"], errors="coerce")
        out["stock_short_name"] = out["stock_short_name"].astype("string")
        out["product_sub_category"] = out["product_sub_category"].astype("string")
        out["asset_class"] = out["asset_class"].astype("string")
        out["geographic_focus"] = out["geographic_focus"].astype("string")
        out["investment_focus"] = out["investment_focus"].astype("string")
        out["management_style"] = out["management_style"].astype("string")
        out["thematic"] = out["thematic"].astype("string")

        numeric_cols = [
            "dividend_yield_pct",
            "ongoing_charges_pct",
            "aum",
            "closing_price_metadata",
            "premium_discount_pct",
        ]
        for col in numeric_cols:
            out[col] = _clean_numeric_series(out[col]).astype("float64")

        out["hkex_code"] = out["stock_code_raw"].apply(_to_hkex_code)
        out["ticker"] = out["stock_code_raw"].apply(_to_yahoo_ticker)
        out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="last")

        instrument_codes = self._load_allowed_instruments()
        out = out[out["hkex_code"].isin(instrument_codes)].copy()

        # Keep sub-category aware AUM treatment first, then fill remaining numeric gaps by mean.
        sub_category_key = out["product_sub_category"].fillna("UNKNOWN")
        group_median = out.groupby(sub_category_key)["aum"].transform("median")
        out["aum"] = out["aum"].fillna(group_median).fillna(out["aum"].mean())

        out["yield_to_cost_ratio"] = out["dividend_yield_pct"] / out["ongoing_charges_pct"].replace(0, np.nan)

        numeric_fill_mean_cols = numeric_cols + ["yield_to_cost_ratio"]
        for col in numeric_fill_mean_cols:
            out[col] = out[col].fillna(out[col].mean())

        keep_cols = [
            "stock_code_raw",
            "stock_short_name",
            "product_sub_category",
            "asset_class",
            "geographic_focus",
            "investment_focus",
            "management_style",
            "thematic",
            "hkex_code",
            "ticker",
            "dividend_yield_pct",
            "ongoing_charges_pct",
            "aum",
            "closing_price_metadata",
            "premium_discount_pct",
            "yield_to_cost_ratio",
        ]
        out = out.loc[:, keep_cols]
        return out

    def _load_allowed_instruments(self) -> set[str]:
        if not self.instruments_path.exists():
            raise FileNotFoundError(f"Instruments file not found: {self.instruments_path}")

        df = pd.read_csv(self.instruments_path)
        if "instruments" not in df.columns:
            raise ValueError(
                f"Expected 'instruments' column in {self.instruments_path}. "
                f"Found: {list(df.columns)}"
            )

        codes = {
            code
            for code in df["instruments"].apply(_to_hkex_code).dropna().tolist()
        }
        if not codes:
            raise ValueError(f"No valid instrument codes parsed from: {self.instruments_path}")
        logger.info("Loaded %s allowed HK ETF instruments from %s", len(codes), self.instruments_path)
        return codes

    def _read_ohlcv_file(self, ticker: str) -> Optional[pd.DataFrame]:
        code = ticker.replace(".HK", "")
        candidate_paths = [
            self.ohlcv_dir / code / "ohlcv.parquet",
            self.ohlcv_dir / f"{code}.parquet",
            self.ohlcv_dir / ticker / "ohlcv.parquet",
            self.ohlcv_dir / f"{ticker}.parquet",
        ]
        file_path = next((path for path in candidate_paths if path.exists()), None)
        if file_path is None:
            return None

        df = pd.read_parquet(file_path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [level0 for level0, *_ in df.columns.tolist()]

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()

        return df

    def _compute_return_risk_features(self, ticker: str) -> dict[str, float]:
        df_ohlcv = self._read_ohlcv_file(ticker)
        if df_ohlcv is None or df_ohlcv.empty or "Close" not in df_ohlcv.columns:
            return {}

        close = pd.to_numeric(df_ohlcv["Close"], errors="coerce").dropna()
        if len(close) < self.rolling_window + 1:
            return {}

        log_returns = np.log(close / close.shift(1)).dropna()
        rolling_vol = log_returns.rolling(self.rolling_window).std()
        vol_30d = float(rolling_vol.iloc[-1]) if not rolling_vol.dropna().empty else np.nan
        annualized_vol = vol_30d * math.sqrt(TRADING_DAYS_PER_YEAR) if pd.notna(vol_30d) else np.nan
        mean_daily_return = float(log_returns.mean()) if not log_returns.empty else np.nan

        sharpe_ratio = np.nan
        if pd.notna(vol_30d) and vol_30d > 0 and pd.notna(mean_daily_return):
            sharpe_ratio = mean_daily_return / vol_30d

        mean_volume = pd.to_numeric(df_ohlcv["Volume"], errors="coerce").dropna().mean()

        return {
            "latest_close": float(close.iloc[-1]),
            "obs_count": float(len(close)),
            "mean_daily_log_return": mean_daily_return,
            "volatility_30d": vol_30d,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "avg_volume": float(mean_volume) if pd.notna(mean_volume) else np.nan,
        }

    def run(self) -> pd.DataFrame:
        metadata = self._load_metadata()
        feature_rows: list[dict[str, float | str]] = []

        logger.info("Engineering OHLCV features for %s ETFs", len(metadata))
        for _, row in metadata.iterrows():
            ticker = row["ticker"]
            metrics = self._compute_return_risk_features(ticker)
            if metrics:
                feature_rows.append({"ticker": ticker, **metrics})

        features_df = pd.DataFrame(feature_rows)
        result = metadata.merge(features_df, on="ticker", how="left")

        # Enforce numeric dtypes and mean-impute numeric gaps in final output.
        numeric_cols = result.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")
            result[col] = result[col].fillna(result[col].mean())

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(self.output_path, index=False)
        logger.info("Saved Financial DNA features to %s", self.output_path)
        logger.info("Output shape: %s rows x %s cols", result.shape[0], result.shape[1])
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Financial DNA features from metadata + OHLCV parquet.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Path to ETP_Data_Export.xlsx")
    parser.add_argument("--ohlcv-dir", type=Path, default=None, help="Directory containing OHLCV parquet files")
    parser.add_argument("--output-path", type=Path, default=None, help="Output parquet path")
    parser.add_argument(
        "--instruments-path",
        type=Path,
        default=None,
        help="Path to all_hk_etf.csv (must include 'instruments' column)",
    )
    parser.add_argument("--rolling-window", type=int, default=30, help="Rolling window for volatility")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    processor = ETFDataProcessor(
        metadata_path=args.metadata_path,
        ohlcv_dir=args.ohlcv_dir,
        output_path=args.output_path,
        instruments_path=args.instruments_path,
        rolling_window=args.rolling_window,
    )
    processor.run()


if __name__ == "__main__":
    main()
