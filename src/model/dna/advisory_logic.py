"""
Advisory logic for Financial DNA cluster outputs.

Module 3 of the 3-module system:
- Home-bias detection
- Hidden twin finder
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_etf_data_root() -> Path:
    data_root = _project_root() / "data"
    return data_root / "etf" if (data_root / "etf").exists() else data_root / "ETF"


class GlobalNavigator:
    """Portfolio advisory layer on top of cluster perspectives."""

    def __init__(
        self,
        clusters_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        min_label_mismatches: int = 2,
        max_pc_distance: float = 2.0,
        top_k_per_etf: int = 10,
        home_bias_max_pc_distance: float = 2.0,
        home_bias_top_k_per_etf: int = 10,
    ) -> None:
        etf_root = _default_etf_data_root()
        self.clusters_path = clusters_path or etf_root / "processed" / "cluster_views" / "cluster_perspectives.parquet"
        self.output_dir = output_dir or etf_root / "processed" / "advisory"
        self.min_label_mismatches = min_label_mismatches
        self.max_pc_distance = max_pc_distance
        self.top_k_per_etf = top_k_per_etf
        self.home_bias_max_pc_distance = home_bias_max_pc_distance
        self.home_bias_top_k_per_etf = home_bias_top_k_per_etf

    def load_clusters(self) -> pd.DataFrame:
        logger.info("Loading cluster perspectives from: %s", self.clusters_path)
        df = pd.read_parquet(self.clusters_path)
        required = ["ticker", "perspective", "cluster_id"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Cluster file missing required columns: {missing}")
        return df

    @staticmethod
    def _safe_str(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def detect_home_bias(self, df_clusters: pd.DataFrame) -> pd.DataFrame:
        """
        Flag ETFs that have mathematically similar alternatives from
        different geographic focus in the same cluster perspective.
        """
        if "geographic_focus" not in df_clusters.columns:
            logger.warning("No 'geographic_focus' column found; home-bias detection will be empty.")
            return pd.DataFrame(
                columns=[
                    "source_ticker",
                    "source_name",
                    "source_geographic_focus",
                    "alternative_ticker",
                    "alternative_name",
                    "alternative_geographic_focus",
                    "perspective",
                    "cluster_id",
                    "signal",
                ]
            )

        rows: list[dict[str, object]] = []
        grouped = df_clusters.groupby(["perspective", "cluster_id"], dropna=False)
        pc_cols = sorted([col for col in df_clusters.columns if col.startswith("pc")])

        for (perspective, cluster_id), group in grouped:
            group = group.copy()
            group["geographic_focus"] = group["geographic_focus"].apply(self._safe_str)

            for idx, left in group.iterrows():
                left_geo = left["geographic_focus"]
                left_ticker = left.get("ticker", "")
                left_name = left.get("stock_short_name", "")

                alternatives = group[
                    (group.index != idx)
                    & (group["geographic_focus"] != "")
                    & (group["geographic_focus"] != left_geo)
                ]
                for _, right in alternatives.iterrows():
                    pc_distance = math.nan
                    if pc_cols:
                        left_vec = pd.to_numeric(left[pc_cols], errors="coerce")
                        right_vec = pd.to_numeric(right[pc_cols], errors="coerce")
                        diff = left_vec.values - right_vec.values
                        if pd.notna(diff).all():
                            pc_distance = float((diff**2).sum() ** 0.5)

                    if pd.notna(pc_distance) and pc_distance <= self.home_bias_max_pc_distance:
                        rows.append(
                            {
                                "source_ticker": left_ticker,
                                "source_name": left_name,
                                "source_geographic_focus": left_geo,
                                "alternative_ticker": right.get("ticker", ""),
                                "alternative_name": right.get("stock_short_name", ""),
                                "alternative_geographic_focus": right.get("geographic_focus", ""),
                                "perspective": perspective,
                                "cluster_id": int(cluster_id),
                                "pc_distance": pc_distance,
                                "signal": "home_bias_candidate",
                            }
                        )

        out = pd.DataFrame(rows).drop_duplicates()
        if not out.empty:
            out = out.sort_values(
                ["source_ticker", "perspective", "cluster_id", "pc_distance"],
                ascending=[True, True, True, True],
            ).reset_index(drop=True)

            kept_rows = []
            per_ticker_count: dict[str, int] = {}
            for _, row in out.iterrows():
                source = self._safe_str(row["source_ticker"])
                alternative = self._safe_str(row["alternative_ticker"])
                if per_ticker_count.get(source, 0) >= self.home_bias_top_k_per_etf:
                    continue
                if per_ticker_count.get(alternative, 0) >= self.home_bias_top_k_per_etf:
                    continue
                kept_rows.append(row.to_dict())
                per_ticker_count[source] = per_ticker_count.get(source, 0) + 1
                per_ticker_count[alternative] = per_ticker_count.get(alternative, 0) + 1
            out = pd.DataFrame(kept_rows)

        if out.empty:
            return pd.DataFrame(
                columns=[
                    "source_ticker",
                    "source_name",
                    "source_geographic_focus",
                    "alternative_ticker",
                    "alternative_name",
                    "alternative_geographic_focus",
                    "perspective",
                    "cluster_id",
                    "pc_distance",
                    "signal",
                ]
            )
        return out.sort_values(["source_ticker", "perspective", "cluster_id"]).reset_index(drop=True)

    def find_hidden_twins(self, df_clusters: pd.DataFrame) -> pd.DataFrame:
        """
        Find ETFs in the same cluster with different labels, indicating
        potential false diversification.
        """
        label_cols = [col for col in ["thematic", "investment_focus", "asset_class"] if col in df_clusters.columns]
        if not label_cols:
            logger.warning("No label columns found for hidden twin detection.")
            return pd.DataFrame(
                columns=[
                    "ticker_a",
                    "name_a",
                    "ticker_b",
                    "name_b",
                    "perspective",
                    "cluster_id",
                    "label_mismatch",
                    "signal",
                ]
            )

        rows: list[dict[str, object]] = []
        grouped = df_clusters.groupby(["perspective", "cluster_id"], dropna=False)
        pc_cols = sorted([col for col in df_clusters.columns if col.startswith("pc")])

        for (perspective, cluster_id), group in grouped:
            group = group.reset_index(drop=True)
            for i in range(len(group)):
                left = group.iloc[i]
                for j in range(i + 1, len(group)):
                    right = group.iloc[j]

                    mismatches = []
                    for label_col in label_cols:
                        left_val = self._safe_str(left.get(label_col, ""))
                        right_val = self._safe_str(right.get(label_col, ""))
                        if left_val and right_val and left_val != right_val:
                            mismatches.append(f"{label_col}:{left_val}!={right_val}")

                    mismatch_count = len(mismatches)
                    if mismatch_count < self.min_label_mismatches:
                        continue

                    pc_distance = math.nan
                    if pc_cols:
                        left_vec = pd.to_numeric(left[pc_cols], errors="coerce")
                        right_vec = pd.to_numeric(right[pc_cols], errors="coerce")
                        diff = left_vec.values - right_vec.values
                        if pd.notna(diff).all():
                            pc_distance = float((diff**2).sum() ** 0.5)

                    if pd.notna(pc_distance) and pc_distance <= self.max_pc_distance:
                        rows.append(
                            {
                                "ticker_a": left.get("ticker", ""),
                                "name_a": left.get("stock_short_name", ""),
                                "ticker_b": right.get("ticker", ""),
                                "name_b": right.get("stock_short_name", ""),
                                "perspective": perspective,
                                "cluster_id": int(cluster_id),
                                "label_mismatch": "|".join(mismatches),
                                "mismatch_count": mismatch_count,
                                "pc_distance": pc_distance,
                                "signal": "hidden_twin_candidate",
                            }
                        )

        out = pd.DataFrame(rows).drop_duplicates()
        if not out.empty:
            out = out.sort_values(
                ["perspective", "cluster_id", "pc_distance", "mismatch_count"],
                ascending=[True, True, True, False],
            ).reset_index(drop=True)

            # Keep only nearest high-confidence pairs per ETF to control row explosion.
            kept_rows = []
            per_ticker_count: dict[str, int] = {}
            for _, row in out.iterrows():
                left = self._safe_str(row["ticker_a"])
                right = self._safe_str(row["ticker_b"])
                if per_ticker_count.get(left, 0) >= self.top_k_per_etf:
                    continue
                if per_ticker_count.get(right, 0) >= self.top_k_per_etf:
                    continue
                kept_rows.append(row.to_dict())
                per_ticker_count[left] = per_ticker_count.get(left, 0) + 1
                per_ticker_count[right] = per_ticker_count.get(right, 0) + 1
            out = pd.DataFrame(kept_rows)

        if out.empty:
            return pd.DataFrame(
                columns=[
                    "ticker_a",
                    "name_a",
                    "ticker_b",
                    "name_b",
                    "perspective",
                    "cluster_id",
                    "label_mismatch",
                    "mismatch_count",
                    "pc_distance",
                    "signal",
                ]
            )
        return out.sort_values(["perspective", "cluster_id", "ticker_a", "ticker_b"]).reset_index(drop=True)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_clusters = self.load_clusters()
        home_bias = self.detect_home_bias(df_clusters)
        hidden_twins = self.find_hidden_twins(df_clusters)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        home_bias_path = self.output_dir / "home_bias_candidates.parquet"
        hidden_twins_path = self.output_dir / "hidden_twin_candidates.parquet"
        home_bias.to_parquet(home_bias_path, index=False)
        hidden_twins.to_parquet(hidden_twins_path, index=False)

        logger.info("Saved home-bias candidates: %s", home_bias_path)
        logger.info("Saved hidden twin candidates: %s", hidden_twins_path)
        logger.info("Home-bias rows: %s | Hidden twin rows: %s", len(home_bias), len(hidden_twins))
        return home_bias, hidden_twins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Financial DNA advisory logic from cluster perspectives.")
    parser.add_argument("--clusters-path", type=Path, default=None, help="Path to cluster_perspectives.parquet")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for advisory parquet files")
    parser.add_argument(
        "--min-label-mismatches",
        type=int,
        default=2,
        help="Minimum number of label mismatches required for hidden twin candidate",
    )
    parser.add_argument(
        "--max-pc-distance",
        type=float,
        default=2.0,
        help="Maximum Euclidean distance in PCA space for hidden twin candidate",
    )
    parser.add_argument(
        "--top-k-per-etf",
        type=int,
        default=10,
        help="Maximum hidden twin pairs to keep per ETF",
    )
    parser.add_argument(
        "--home-bias-max-pc-distance",
        type=float,
        default=2.0,
        help="Maximum Euclidean distance in PCA space for home-bias candidates",
    )
    parser.add_argument(
        "--home-bias-top-k-per-etf",
        type=int,
        default=10,
        help="Maximum home-bias pairs to keep per ETF",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    navigator = GlobalNavigator(
        clusters_path=args.clusters_path,
        output_dir=args.output_dir,
        min_label_mismatches=args.min_label_mismatches,
        max_pc_distance=args.max_pc_distance,
        top_k_per_etf=args.top_k_per_etf,
        home_bias_max_pc_distance=args.home_bias_max_pc_distance,
        home_bias_top_k_per_etf=args.home_bias_top_k_per_etf,
    )
    navigator.run()


if __name__ == "__main__":
    main()
