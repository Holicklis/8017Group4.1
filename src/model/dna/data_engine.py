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
TRADING_DAYS_PER_MONTH = 21
BENCHMARK_SYMBOLS = [
    "^AXJO",
    "^HSI",
    "^KS11",
    "^MXWO",
    "^N225",
    "^SPX",
    "^STOXX50E",
    "^TWII",
    "XAU",
    "^SPGSCL",
    "BTC",
]
REQUIRED_METADATA_COLUMNS = [
    "Stock code*",
    "Stock short name*",
    "Listing date*",
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


def _default_model_output_root() -> Path:
    return _project_root() / "model_output" / "dna"


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


def _symbol_to_file_token(symbol: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]", "", symbol.upper())
    return token


class ETFDataProcessor:
    """Build Financial DNA features from metadata and OHLCV parquet files."""

    def __init__(
        self,
        metadata_path: Optional[Path] = None,
        ohlcv_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        instruments_path: Optional[Path] = None,
        benchmark_dir: Optional[Path] = None,
        holdings_dir: Optional[Path] = None,
        rolling_window: int = 30,
        min_price_points: int = 200,
        macro_corr_window_days: int = TRADING_DAYS_PER_YEAR * 3,
        macro_min_overlap_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        etf_root = _default_etf_data_root()
        model_output_root = _default_model_output_root()
        self.metadata_path = metadata_path or etf_root / "summary" / "ETP_Data_Export.xlsx"
        self.ohlcv_dir = ohlcv_dir or etf_root / "ohlcv"
        self.output_path = output_path or model_output_root / "financial_dna.parquet"
        self.instruments_path = instruments_path or etf_root / "instruments" / "all_hk_etf.csv"
        self.benchmark_dir = benchmark_dir or self.ohlcv_dir
        self.holdings_dir = holdings_dir or etf_root / "holdings" / "top10"
        self.rolling_window = rolling_window
        self.min_price_points = min_price_points
        self.macro_corr_window_days = macro_corr_window_days
        self.macro_min_overlap_days = macro_min_overlap_days
        self._benchmark_cache: dict[str, pd.Series] = {}

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
                "Listing date*": "listing_date",
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
        out["listing_date"] = pd.to_datetime(out["listing_date"], errors="coerce")
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
        out = out[out["listing_date"].isna() | (out["listing_date"] <= pd.Timestamp("2024-12-31"))].copy()
        # out = out[out["asset_class"].str.lower().str.contains("equity", na=False)].copy()

        # Keep sub-category aware AUM treatment first, then fill remaining numeric gaps by mean.
        sub_category_key = out["product_sub_category"].fillna("UNKNOWN")
        group_median = out.groupby(sub_category_key)["aum"].transform("median")
        out["aum"] = out["aum"].fillna(group_median).fillna(out["aum"].mean())

        # Business rule: fund yield missing values should be treated as 0.
        out["dividend_yield_pct"] = out["dividend_yield_pct"].fillna(0.0)
        out["yield_to_cost_ratio"] = out["dividend_yield_pct"] / out["ongoing_charges_pct"].replace(0, np.nan)
        out["concentration_top10"] = out["hkex_code"].apply(self._compute_concentration_top10)

        numeric_fill_mean_cols = [col for col in numeric_cols + ["yield_to_cost_ratio"] if col != "dividend_yield_pct"]
        for col in numeric_fill_mean_cols:
            out[col] = out[col].fillna(out[col].mean())

        keep_cols = [
            "stock_code_raw",
            "stock_short_name",
            "listing_date",
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
            "premium_discount_pct",
            "yield_to_cost_ratio",
            "concentration_top10",
        ]
        out = out.loc[:, keep_cols]
        return out

    def _compute_concentration_top10(self, hkex_code: str) -> float:
        if not hkex_code:
            return 0.9

        candidate_dirs = [
            self.holdings_dir / hkex_code,
            self.holdings_dir / str(int(hkex_code)),
        ]
        candidate_files: list[Path] = []
        for folder in candidate_dirs:
            candidate_files.extend([folder / "top_holdings.parquet", folder / "top_holdings.csv"])

        file_path = next((p for p in candidate_files if p.exists()), None)
        if file_path is None:
            return 0.9

        df_holdings = pd.read_parquet(file_path) if file_path.suffix == ".parquet" else pd.read_csv(file_path)
        if df_holdings.empty:
            return 0.9

        # Try common weight column names first.
        preferred_cols = [
            "holdingPercent",
            "holding_percent",
            "holding_pct",
            "weight",
            "weight_pct",
            "percent",
            "percentage",
        ]
        weight_col = next((col for col in preferred_cols if col in df_holdings.columns), None)
        if weight_col is None:
            # Fallback: detect first numeric-like column that looks like a weight.
            for col in df_holdings.columns:
                col_lower = str(col).lower()
                if any(token in col_lower for token in ["percent", "weight", "holding"]):
                    weight_col = col
                    break
        if weight_col is None:
            return 0.9

        weights = _clean_numeric_series(df_holdings[weight_col]).dropna().head(10)
        if weights.empty:
            return 0.9

        concentration = float(weights.sum())
        # Normalize if values are stored in 0-100 range.
        if concentration > 1.5:
            concentration = concentration / 100.0
        return float(np.clip(concentration, 0.0, 1.0))

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
        if len(close) < max(self.rolling_window + 1, self.min_price_points):
            return {}

        log_returns = np.log(close / close.shift(1)).dropna()
        monthly_returns = close.resample("ME").last().pct_change().dropna()
        rolling_vol = log_returns.rolling(self.rolling_window).std()
        vol_30d = float(rolling_vol.iloc[-1]) if not rolling_vol.dropna().empty else np.nan
        annualized_vol = vol_30d * math.sqrt(TRADING_DAYS_PER_YEAR) if pd.notna(vol_30d) else np.nan
        mean_daily_return = float(log_returns.mean()) if not log_returns.empty else np.nan

        ret_1y = self._period_return(close, periods=TRADING_DAYS_PER_YEAR)
        ret_3y = self._period_return(close, periods=TRADING_DAYS_PER_YEAR * 3)
        ret_5y = self._period_return(close, periods=TRADING_DAYS_PER_YEAR * 5)
        yearly_candidates = [
            ret_1y,
            ret_3y / 3 if pd.notna(ret_3y) else np.nan,
            ret_5y / 5 if pd.notna(ret_5y) else np.nan,
        ]
        valid_candidates = [x for x in yearly_candidates if pd.notna(x)]
        avg_yearly_return = float(np.mean(valid_candidates)) if valid_candidates else np.nan
        monthly_vol = float(monthly_returns.std()) if not monthly_returns.empty else np.nan

        sharpe_window = min(len(log_returns), TRADING_DAYS_PER_YEAR * 3)
        sharpe_returns = log_returns.tail(sharpe_window)
        sharpe_ratio = np.nan
        sharpe_vol = float(sharpe_returns.std()) if not sharpe_returns.empty else np.nan
        sharpe_mean = float(sharpe_returns.mean()) if not sharpe_returns.empty else np.nan
        if pd.notna(sharpe_vol) and sharpe_vol > 0 and pd.notna(sharpe_mean):
            sharpe_ratio = (sharpe_mean / sharpe_vol) * math.sqrt(TRADING_DAYS_PER_YEAR)

        macro_corrs = self._compute_macro_correlations(log_returns)

        features: dict[str, float] = {
            "obs_count": float(len(close)),
            "mean_daily_log_return": mean_daily_return,
            "return_1y": ret_1y,
            "return_3y": ret_3y,
            "return_5y": ret_5y,
            "average_yearly_return": float(avg_yearly_return) if pd.notna(avg_yearly_return) else np.nan,
            "monthly_volatility": monthly_vol,
            "volatility_30d": vol_30d,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "return_to_risk_1y": (ret_1y / annualized_vol) if pd.notna(ret_1y) and pd.notna(annualized_vol) and annualized_vol > 0 else np.nan,
        }
        features.update(macro_corrs)
        return features

    @staticmethod
    def _period_return(close: pd.Series, periods: int) -> float:
        if len(close) <= periods:
            return np.nan
        start = float(close.iloc[-(periods + 1)])
        end = float(close.iloc[-1])
        if start <= 0:
            return np.nan
        return (end / start) - 1.0

    def _load_benchmark_returns(self, symbol: str) -> Optional[pd.Series]:
        if symbol in self._benchmark_cache:
            return self._benchmark_cache[symbol]

        token = _symbol_to_file_token(symbol)
        candidate_paths = [
            self.benchmark_dir / f"{symbol}.parquet",
            self.benchmark_dir / f"{token}.parquet",
            self.benchmark_dir / symbol / "ohlcv.parquet",
            self.benchmark_dir / token / "ohlcv.parquet",
            self.benchmark_dir / f"{symbol}.csv",
            self.benchmark_dir / f"{token}.csv",
            self.benchmark_dir / symbol / "ohlcv.csv",
            self.benchmark_dir / token / "ohlcv.csv",
        ]
        file_path = next((path for path in candidate_paths if path.exists()), None)
        if file_path is None:
            self._benchmark_cache[symbol] = None
            return None

        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()

        close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if close_col is None:
            self._benchmark_cache[symbol] = None
            return None

        close = pd.to_numeric(df[close_col], errors="coerce").dropna()
        if close.empty:
            self._benchmark_cache[symbol] = None
            return None
        returns = np.log(close / close.shift(1)).dropna()
        self._benchmark_cache[symbol] = returns
        return returns

    def _compute_macro_correlations(self, etf_log_returns: pd.Series) -> dict[str, float]:
        out: dict[str, float] = {}
        etf_returns = etf_log_returns.sort_index()
        for symbol in BENCHMARK_SYMBOLS:
            bench_returns = self._load_benchmark_returns(symbol)
            token = _symbol_to_file_token(symbol).lower()
            corr_feature_name = f"corr_{token}"
            beta_feature_name = f"beta_{token}"
            if bench_returns is None:
                out[corr_feature_name] = np.nan
                out[beta_feature_name] = np.nan
                continue

            # Reindex benchmark to ETF calendar first for consistent overlap handling.
            bench_aligned = bench_returns.reindex(etf_returns.index)
            aligned = pd.concat([etf_returns, bench_aligned], axis=1).dropna()
            if len(aligned) < self.macro_min_overlap_days:
                out[corr_feature_name] = np.nan
                out[beta_feature_name] = np.nan
                continue

            # Use a fixed trailing window for more comparable correlations.
            aligned = aligned.tail(self.macro_corr_window_days)
            if len(aligned) < self.macro_min_overlap_days:
                out[corr_feature_name] = np.nan
                out[beta_feature_name] = np.nan
                continue

            etf_series = aligned.iloc[:, 0]
            bench_series = aligned.iloc[:, 1]
            out[corr_feature_name] = float(etf_series.corr(bench_series))

            bench_var = float(bench_series.var())
            if pd.notna(bench_var) and bench_var > 0:
                covariance = float(etf_series.cov(bench_series))
                out[beta_feature_name] = covariance / bench_var
            else:
                out[beta_feature_name] = np.nan
        return out

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
            if col == "dividend_yield_pct":
                result[col] = result[col].fillna(0.0)
            else:
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
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Directory containing macro benchmark OHLCV files (.parquet/.csv)",
    )
    parser.add_argument(
        "--holdings-dir",
        type=Path,
        default=None,
        help="Directory containing ETF top-10 holdings files",
    )
    parser.add_argument(
        "--macro-corr-window-days",
        type=int,
        default=TRADING_DAYS_PER_YEAR * 3,
        help="Trailing window size (in trading days) for macro correlations",
    )
    parser.add_argument(
        "--macro-min-overlap-days",
        type=int,
        default=TRADING_DAYS_PER_YEAR,
        help="Minimum overlapping trading days required for macro correlation",
    )
    parser.add_argument("--rolling-window", type=int, default=30, help="Rolling window for volatility")
    parser.add_argument(
        "--min-price-points",
        type=int,
        default=200,
        help="Minimum number of valid close-price observations required per ETF",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    processor = ETFDataProcessor(
        metadata_path=args.metadata_path,
        ohlcv_dir=args.ohlcv_dir,
        output_path=args.output_path,
        instruments_path=args.instruments_path,
        benchmark_dir=args.benchmark_dir,
        holdings_dir=args.holdings_dir,
        rolling_window=args.rolling_window,
        min_price_points=args.min_price_points,
        macro_corr_window_days=args.macro_corr_window_days,
        macro_min_overlap_days=args.macro_min_overlap_days,
    )
    processor.run()


if __name__ == "__main__":
    main()
