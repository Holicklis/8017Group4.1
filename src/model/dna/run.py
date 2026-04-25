from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from advisory_logic import GlobalNavigator
from data_engine import ETFDataProcessor
from model_core import MultiClusterPCAEngine
from visualize_clusters import ClusterVisualizer


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    return _project_root() / "model_output" / "dna"


def run_dna_pipeline(
    output_root: Path | None = None,
    skip_visualization: bool = False,
) -> dict[str, str]:
    root = output_root or _default_output_root()
    root.mkdir(parents=True, exist_ok=True)

    # Step 1: feature engineering
    processor = ETFDataProcessor(output_path=root / "financial_dna.parquet")
    processor.run()

    # Step 2: clustering
    cluster_dir = root / "cluster_views"
    clustering = MultiClusterPCAEngine(
        input_path=root / "financial_dna.parquet",
        output_dir=cluster_dir,
    )
    combined, by_perspective = clustering.run_all_perspectives()
    clustering.save_outputs(combined=combined, by_perspective=by_perspective)

    # Step 3: advisory
    advisory_dir = root / "advisory"
    navigator = GlobalNavigator(
        clusters_path=cluster_dir / "cluster_perspectives.parquet",
        output_dir=advisory_dir,
    )
    home_bias_df, hidden_twin_df = navigator.run()

    # Step 4: visualization
    plot_dir = cluster_dir / "plots"
    if not skip_visualization:
        visualizer = ClusterVisualizer(
            input_path=cluster_dir / "cluster_perspectives.parquet",
            output_dir=plot_dir,
        )
        visualizer.run()

    summary = {
        "output_root": str(root),
        "financial_dna_parquet": str(root / "financial_dna.parquet"),
        "cluster_perspectives_parquet": str(cluster_dir / "cluster_perspectives.parquet"),
        "cluster_views_dir": str(cluster_dir),
        "advisory_dir": str(advisory_dir),
        "plots_dir": str(plot_dir),
        "home_bias_rows": str(len(home_bias_df)),
        "hidden_twin_rows": str(len(hidden_twin_df)),
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = root / "dna_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Financial DNA model pipeline.")
    parser.add_argument("--output-root", type=Path, default=None, help="Root output directory for DNA artifacts")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip plot generation stage")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()
    result = run_dna_pipeline(output_root=args.output_root, skip_visualization=args.skip_visualization)
    print(json.dumps(result, indent=2))
