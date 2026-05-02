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


def _default_model_output_root() -> Path:
    return _project_root() / "model_output" / "dna"


class ClusterVisualizer:
    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        annotate_points: bool = False,
        plotly_html: bool = True,
        show_point_text: bool = False,
        show_cluster_centroids: bool = True,
        fig_width: float = 16.0,
        fig_height: float = 9.0,
        title_fontsize: int = 20,
        axis_label_fontsize: int = 20,
        tick_fontsize: int = 14,
        legend_fontsize: int = 14,
        marker_size: float = 95.0,
        output_dpi: int = 300,
    ) -> None:
        model_output_root = _default_model_output_root()
        self.input_path = input_path or model_output_root / "cluster_views" / "cluster_perspectives.parquet"
        self.output_dir = output_dir or model_output_root / "cluster_views" / "plots"
        self.annotate_points = annotate_points
        self.plotly_html = plotly_html
        self.show_point_text = show_point_text
        self.show_cluster_centroids = show_cluster_centroids
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.title_fontsize = title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.tick_fontsize = tick_fontsize
        self.legend_fontsize = legend_fontsize
        self.marker_size = marker_size
        self.output_dpi = output_dpi

    def _load_clusters(self) -> pd.DataFrame:
        logger.info("Loading cluster perspectives: %s", self.input_path)
        df = pd.read_parquet(self.input_path)
        required_cols = ["perspective", "cluster_id", "pc1", "pc2"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in cluster file: {missing}")
        return df

    def _plot_scatter(self, df: pd.DataFrame, perspective: str, x_col: str, y_col: str) -> None:
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        clusters = sorted(df["cluster_id"].dropna().unique().tolist())
        cmap = plt.get_cmap("tab10")

        for idx, cluster_id in enumerate(clusters):
            chunk = df[df["cluster_id"] == cluster_id]
            ax.scatter(
                chunk[x_col],
                chunk[y_col],
                s=self.marker_size,
                alpha=0.9,
                color=cmap(idx % 10),
                label=f"Cluster {int(cluster_id)}",
                edgecolors="white",
                linewidths=0.5,
            )

            if self.annotate_points and "ticker" in chunk.columns:
                for _, row in chunk.iterrows():
                    ax.annotate(
                        str(row["ticker"]),
                        (row[x_col], row[y_col]),
                        fontsize=7,
                        alpha=0.8,
                    )

        if perspective == "return_risk_profile":
            plot_title = "Risk / Return Cluster"
        elif perspective == "macro_sensitivity":
            plot_title = "Macro Sensitivity Cluster"
        else:
            plot_title = perspective
        ax.set_title(plot_title, fontsize=self.title_fontsize, pad=12, weight="bold")
        ax.set_xlabel(x_col.upper(), fontsize=self.axis_label_fontsize, labelpad=10)
        ax.set_ylabel(y_col.upper(), fontsize=self.axis_label_fontsize, labelpad=10)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="best",
            fontsize=self.legend_fontsize,
            title="Clusters",
            title_fontsize=max(self.legend_fontsize, self.tick_fontsize),
            frameon=True,
        )
        fig.tight_layout()

        if x_col == "pc1" and y_col == "pc2":
            output_path = self.output_dir / f"{perspective}_pc_scatter.png"
        else:
            output_path = self.output_dir / f"{perspective}_{x_col}_{y_col}_scatter.png"
        fig.savefig(output_path, dpi=self.output_dpi)
        plt.close(fig)
        logger.info("Saved scatter plot: %s", output_path)

    def _plot_scatter_plotly(self, df: pd.DataFrame, perspective: str, x_col: str, y_col: str) -> None:
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
            x=x_col,
            y=y_col,
            color=df_plot["cluster_id"].astype(str),
            hover_data={
                "ticker": True,
                "stock_short_name": True,
                "geographic_focus": True,
                "cluster_id": True,
                x_col: ":.4f",
                y_col: ":.4f",
            },
            title=f"Financial DNA Cluster Map - {perspective}",
            labels={"color": "Cluster ID", x_col: x_col.upper(), y_col: y_col.upper()},
        )
        if self.show_point_text:
            fig.update_traces(text=df_plot["label_text"], textposition="top center")

        if self.show_cluster_centroids:
            centroids = (
                df_plot.groupby("cluster_id", dropna=False)[[x_col, y_col]]
                .mean()
                .reset_index()
                .sort_values("cluster_id")
            )
            fig.add_scatter(
                x=centroids[x_col],
                y=centroids[y_col],
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

        if x_col == "pc1" and y_col == "pc2":
            html_path = self.output_dir / f"{perspective}_pc_scatter.html"
        else:
            html_path = self.output_dir / f"{perspective}_{x_col}_{y_col}_scatter.html"
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
            self._plot_scatter(chunk.copy(), perspective=perspective, x_col="pc1", y_col="pc2")
            if self.plotly_html:
                self._plot_scatter_plotly(chunk.copy(), perspective=perspective, x_col="pc1", y_col="pc2")

        self._save_cluster_size_summary(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Financial DNA cluster perspectives.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Path to cluster_perspectives.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated plot files",
    )
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
    parser.add_argument("--fig-width", type=float, default=16.0, help="Matplotlib figure width in inches")
    parser.add_argument("--fig-height", type=float, default=9.0, help="Matplotlib figure height in inches")
    parser.add_argument("--title-fontsize", type=int, default=20, help="Plot title font size")
    parser.add_argument("--axis-label-fontsize", type=int, default=20, help="Axis label font size")
    parser.add_argument("--tick-fontsize", type=int, default=14, help="Axis tick font size")
    parser.add_argument("--legend-fontsize", type=int, default=14, help="Legend font size")
    parser.add_argument("--marker-size", type=float, default=95.0, help="Scatter marker size")
    parser.add_argument("--output-dpi", type=int, default=300, help="PNG output DPI")
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
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        title_fontsize=args.title_fontsize,
        axis_label_fontsize=args.axis_label_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        marker_size=args.marker_size,
        output_dpi=args.output_dpi,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
