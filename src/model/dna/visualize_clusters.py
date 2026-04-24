"""
Visualization utilities for Financial DNA cluster outputs.

Generates:
- 2D PCA scatter plots by perspective (pc1 vs pc2)
- Cluster size summary CSV and bar charts
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_etf_data_root() -> Path:
    data_root = _project_root() / "data"
    return data_root / "etf" if (data_root / "etf").exists() else data_root / "ETF"


class ClusterVisualizer:
    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        annotate_points: bool = False,
        plotly_html: bool = True,
        show_point_text: bool = False,
        show_cluster_centroids: bool = True,
    ) -> None:
        etf_root = _default_etf_data_root()
        self.input_path = input_path or etf_root / "processed" / "cluster_views" / "cluster_perspectives.parquet"
        self.output_dir = output_dir or etf_root / "processed" / "cluster_views" / "plots"
        self.annotate_points = annotate_points
        self.plotly_html = plotly_html
        self.show_point_text = show_point_text
        self.show_cluster_centroids = show_cluster_centroids

    def _load_clusters(self) -> pd.DataFrame:
        logger.info("Loading cluster perspectives: %s", self.input_path)
        df = pd.read_parquet(self.input_path)
        required_cols = ["perspective", "cluster_id", "pc1", "pc2"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in cluster file: {missing}")
        return df

    def _plot_scatter(self, df: pd.DataFrame, perspective: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 7))
        clusters = sorted(df["cluster_id"].dropna().unique().tolist())
        cmap = plt.get_cmap("tab10")

        for idx, cluster_id in enumerate(clusters):
            chunk = df[df["cluster_id"] == cluster_id]
            ax.scatter(
                chunk["pc1"],
                chunk["pc2"],
                s=35,
                alpha=0.85,
                color=cmap(idx % 10),
                label=f"Cluster {int(cluster_id)}",
            )

            if self.annotate_points and "ticker" in chunk.columns:
                for _, row in chunk.iterrows():
                    ax.annotate(str(row["ticker"]), (row["pc1"], row["pc2"]), fontsize=7, alpha=0.8)

        ax.set_title(f"Financial DNA Cluster Map - {perspective}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        output_path = self.output_dir / f"{perspective}_pc_scatter.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        logger.info("Saved scatter plot: %s", output_path)

    def _plot_scatter_plotly(self, df: pd.DataFrame, perspective: str) -> None:
        df_plot = df.copy()
        for col in ["stock_short_name", "geographic_focus", "ticker"]:
            if col not in df_plot.columns:
                df_plot[col] = ""
            df_plot[col] = df_plot[col].fillna("").astype(str)

        # Show ETF name + country directly on points as requested.
        df_plot["label_text"] = (
            df_plot["stock_short_name"].str.strip().replace("", "NA")
            + " ("
            + df_plot["geographic_focus"].str.strip().replace("", "NA")
            + ")"
        )

        fig = px.scatter(
            df_plot,
            x="pc1",
            y="pc2",
            color=df_plot["cluster_id"].astype(str),
            hover_data={
                "ticker": True,
                "stock_short_name": True,
                "geographic_focus": True,
                "cluster_id": True,
                "pc1": ":.4f",
                "pc2": ":.4f",
            },
            title=f"Financial DNA Cluster Map - {perspective}",
            labels={"color": "Cluster ID", "pc1": "PC1", "pc2": "PC2"},
        )
        if self.show_point_text:
            fig.update_traces(text=df_plot["label_text"], textposition="top center")

        if self.show_cluster_centroids:
            centroids = (
                df_plot.groupby("cluster_id", dropna=False)[["pc1", "pc2"]]
                .mean()
                .reset_index()
                .sort_values("cluster_id")
            )
            fig.add_scatter(
                x=centroids["pc1"],
                y=centroids["pc2"],
                mode="markers+text",
                text=[f"Cluster {int(cid)}" for cid in centroids["cluster_id"]],
                textposition="top center",
                marker={"size": 14, "symbol": "x", "color": "black"},
                name="Cluster Center",
                hoverinfo="skip",
            )

        fig.update_traces(marker={"size": 9, "opacity": 0.82})
        fig.update_layout(
            height=760,
            width=1200,
            template="plotly_white",
            legend_title_text="Cluster ID",
        )

        html_path = self.output_dir / f"{perspective}_pc_scatter.html"
        fig.write_html(html_path)
        logger.info("Saved interactive plotly scatter: %s", html_path)

    def _save_cluster_size_summary(self, df: pd.DataFrame) -> None:
        summary = (
            df.groupby(["perspective", "cluster_id"], dropna=False)
            .size()
            .reset_index(name="etf_count")
            .sort_values(["perspective", "cluster_id"])
            .reset_index(drop=True)
        )
        csv_path = self.output_dir / "cluster_size_summary.csv"
        summary.to_csv(csv_path, index=False)
        logger.info("Saved cluster size summary: %s", csv_path)

        for perspective, chunk in summary.groupby("perspective"):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(chunk["cluster_id"].astype(str), chunk["etf_count"])
            ax.set_title(f"Cluster Size Distribution - {perspective}")
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("ETF Count")
            ax.grid(True, axis="y", alpha=0.2)
            fig.tight_layout()
            output_path = self.output_dir / f"{perspective}_cluster_sizes.png"
            fig.savefig(output_path, dpi=180)
            plt.close(fig)
            logger.info("Saved cluster size plot: %s", output_path)

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = self._load_clusters()

        for perspective, chunk in df.groupby("perspective"):
            if chunk["pc1"].notna().sum() == 0 or chunk["pc2"].notna().sum() == 0:
                logger.warning("Skipping %s: pc1/pc2 missing.", perspective)
                continue
            self._plot_scatter(chunk.copy(), perspective=perspective)
            if self.plotly_html:
                self._plot_scatter_plotly(chunk.copy(), perspective=perspective)

        self._save_cluster_size_summary(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Financial DNA cluster perspectives.")
    parser.add_argument("--input-path", type=Path, default=None, help="Path to cluster_perspectives.parquet")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated plot files")
    parser.add_argument(
        "--annotate-points",
        action="store_true",
        help="Annotate scatter points with ticker labels (can be crowded for many ETFs)",
    )
    parser.add_argument(
        "--no-plotly-html",
        action="store_true",
        help="Disable interactive plotly HTML scatter export",
    )
    parser.add_argument(
        "--show-point-text",
        action="store_true",
        help="Show ETF name+country text directly on every point (can be crowded)",
    )
    parser.add_argument(
        "--no-cluster-centroids",
        action="store_true",
        help="Disable centroid markers and labels",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    visualizer = ClusterVisualizer(
        input_path=args.input_path,
        output_dir=args.output_dir,
        annotate_points=args.annotate_points,
        plotly_html=not args.no_plotly_html,
        show_point_text=args.show_point_text,
        show_cluster_centroids=not args.no_cluster_centroids,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
