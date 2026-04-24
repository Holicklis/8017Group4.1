"""
Multi-perspective PCA + clustering engine for Financial DNA.

Module 2 of the 3-module system:
- Standardize selected Financial DNA features
- Reduce dimensions with PCA (target explained variance threshold)
- Cluster ETFs with K-Means for multiple perspectives
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_etf_data_root() -> Path:
    data_root = _project_root() / "data"
    return data_root / "etf" if (data_root / "etf").exists() else data_root / "ETF"


class MultiClusterPCAEngine:
    """Run PCA + KMeans for one or multiple feature perspectives."""

    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        variance_threshold: float = 0.95,
        n_clusters: int = 4,
        random_state: int = 42,
    ) -> None:
        etf_root = _default_etf_data_root()
        self.input_path = input_path or etf_root / "processed" / "financial_dna.parquet"
        self.output_dir = output_dir or etf_root / "processed" / "cluster_views"
        self.variance_threshold = variance_threshold
        self.n_clusters = n_clusters
        self.random_state = random_state

    @staticmethod
    def _default_perspectives() -> dict[str, list[str]]:
        return {
            "risk_return": [
                "mean_daily_log_return",
                "volatility_30d",
                "annualized_volatility",
                "sharpe_ratio",
                "latest_close",
            ],
            "income_efficiency": [
                "dividend_yield_pct",
                "ongoing_charges_pct",
                "yield_to_cost_ratio",
            ],
            "liquidity_scale": [
                "aum",
                "avg_volume",
                "premium_discount_pct",
            ],
            "macro_sensitivity": [
                "beta",
                "corr_hsi",
                "corr_sp500",
                "corr_mxwo",
            ],
        }

    @staticmethod
    def _base_output_columns(df: pd.DataFrame) -> list[str]:
        candidates = [
            "ticker",
            "stock_short_name",
            "geographic_focus",
            "asset_class",
            "investment_focus",
            "management_style",
            "thematic",
        ]
        return [col for col in candidates if col in df.columns]

    @staticmethod
    def _prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
        usable_cols = [col for col in feature_cols if col in df.columns]
        if not usable_cols:
            return pd.DataFrame(index=df.index), []

        x = df.loc[:, usable_cols].copy()
        for col in usable_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].fillna(x[col].mean())

        # Drop columns with all NaN after coercion/mean attempt.
        x = x.dropna(axis=1, how="all")
        usable_cols = x.columns.tolist()

        if usable_cols:
            # Safety pass for any still-missing values.
            for col in usable_cols:
                x[col] = x[col].fillna(x[col].mean())
        return x, usable_cols

    def run_single_perspective(
        self,
        df: pd.DataFrame,
        perspective_name: str,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        x, used_features = self._prepare_feature_matrix(df, feature_cols)
        if x.empty or len(used_features) < 2:
            raise ValueError(
                f"Perspective '{perspective_name}' needs at least 2 usable numeric features. "
                f"Requested: {feature_cols}, usable: {used_features}"
            )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        full_pca = PCA()
        full_pca.fit(x_scaled)
        cumulative = full_pca.explained_variance_ratio_.cumsum()
        n_components = int((cumulative < self.variance_threshold).sum() + 1)
        n_components = min(max(2, n_components), x.shape[1])

        pca = PCA(n_components=n_components, random_state=self.random_state)
        pcs = pca.fit_transform(x_scaled)

        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        clusters = km.fit_predict(pcs)

        output_cols = self._base_output_columns(df)
        result = df.loc[:, output_cols].copy()
        result["perspective"] = perspective_name
        result["cluster_id"] = clusters.astype(int)
        result["n_components"] = n_components
        result["explained_variance"] = float(pca.explained_variance_ratio_.sum())
        result["features_used"] = ",".join(used_features)

        for idx in range(pcs.shape[1]):
            result[f"pc{idx + 1}"] = pcs[:, idx]

        return result

    def run_all_perspectives(
        self,
        perspectives: Optional[dict[str, list[str]]] = None,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        logger.info("Loading Financial DNA: %s", self.input_path)
        df = pd.read_parquet(self.input_path)
        perspectives = perspectives or self._default_perspectives()

        all_results: list[pd.DataFrame] = []
        by_perspective: dict[str, pd.DataFrame] = {}

        for name, features in perspectives.items():
            try:
                result = self.run_single_perspective(df=df, perspective_name=name, feature_cols=features)
                by_perspective[name] = result
                all_results.append(result)
                logger.info(
                    "Built perspective '%s': rows=%s, explained_variance=%.4f",
                    name,
                    len(result),
                    result["explained_variance"].iloc[0],
                )
            except ValueError as exc:
                logger.warning("Skipping perspective '%s': %s", name, exc)

        if not all_results:
            raise ValueError("No perspective could be built. Check feature availability in financial_dna.parquet.")

        combined = pd.concat(all_results, axis=0, ignore_index=True)
        return combined, by_perspective

    def save_outputs(self, combined: pd.DataFrame, by_perspective: dict[str, pd.DataFrame]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        combined_path = self.output_dir / "cluster_perspectives.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info("Saved combined cluster perspectives: %s", combined_path)

        for name, df_view in by_perspective.items():
            path = self.output_dir / f"{name}.parquet"
            df_view.to_parquet(path, index=False)
            logger.info("Saved perspective '%s': %s", name, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA + KMeans cluster perspectives on Financial DNA features.")
    parser.add_argument("--input-path", type=Path, default=None, help="Path to financial_dna.parquet")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for cluster parquet files")
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Target cumulative explained variance for PCA component selection",
    )
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of KMeans clusters per perspective")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    engine = MultiClusterPCAEngine(
        input_path=args.input_path,
        output_dir=args.output_dir,
        variance_threshold=args.variance_threshold,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    combined, by_perspective = engine.run_all_perspectives()
    engine.save_outputs(combined=combined, by_perspective=by_perspective)


if __name__ == "__main__":
    main()
