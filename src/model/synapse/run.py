from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from evaluate_benchmark import _default_output_path, run_side_by_side_evaluation
from run_news_events import run_news_events
from semantic_clustering_stability import run_stability_assessment

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> Path:
    log_dir = _project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "synapse_run.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    has_file_handler = any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file
        for handler in root_logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    return log_file


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_news_csv() -> Path:
    return _project_root() / "data" / "news" / "financial_news_events.csv"


def _default_synapse_output_root() -> Path:
    return _project_root() / "model_output" / "synpse"


def run_synapse_pipeline(
    preset: str = "fast",
    corpus_mode: str = "profile",
    top_k: int = 5,
    run_news: bool = True,
    run_stability: bool = True,
    news_csv: Path | None = None,
) -> dict[str, object]:
    logger.info("Starting Synapse pipeline run")
    output_root = _default_synapse_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "output_root": str(output_root),
        "timestamp": datetime.now().isoformat(),
    }

    # Step 1: benchmark
    benchmark = run_side_by_side_evaluation(
        preset=preset,
        top_k=max(top_k, 3),
        sentence_row_cap=4000,
    )
    benchmark_path = _default_output_path()
    benchmark_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    results["benchmark_json"] = str(benchmark_path)
    logger.info("Benchmark output saved to %s", benchmark_path)

    # Step 2: batch news run
    if run_news:
        input_csv = news_csv or _default_news_csv()
        if input_csv.exists():
            news_result = run_news_events(
                input_csv=input_csv,
                output_dir=None,
                preset=preset,
                corpus_mode=corpus_mode,
                top_k=max(top_k, 3),
                text_col="Headline",
                sentiment_model_name="ProsusAI/finbert",
                sentiment_weight=0.25,
            )
            results["news_run"] = news_result
        else:
            results["news_run"] = {"skipped": f"Input csv not found: {input_csv}"}
            logger.warning("Skipping news run; input CSV not found at %s", input_csv)

    # Step 3: stability run
    if run_stability:
        stability_result = run_stability_assessment(
            output_dir=None,
            preset=preset,
            corpus_mode=corpus_mode,
            top_k=max(top_k, 5),
            num_concepts=30,
            variants_per_concept=10,
            tie_epsilon=0.01,
            use_light_rerank=True,
            cross_top_n=12,
            apply_query_canonicalization=True,
        )
        results["stability_run"] = stability_result

    summary_path = output_root / "synapse_run_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["summary_json"] = str(summary_path)
    logger.info("Synapse pipeline completed. Summary saved to %s", summary_path)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Synapse model pipeline.")
    parser.add_argument("--preset", type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument("--corpus-mode", type=str, default="profile", choices=["profile", "sentence"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--news-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    log_file = configure_logging()
    args = _parse_args()
    output = run_synapse_pipeline(
        preset=args.preset,
        corpus_mode=args.corpus_mode,
        top_k=max(args.top_k, 1),
        run_news=not args.skip_news,
        run_stability=not args.skip_stability,
        news_csv=args.news_csv,
    )
    logger.info("Run completed. summary_json=%s", output.get("summary_json"))
    logger.debug("Run payload: %s", json.dumps(output, indent=2))
    logger.info("Logs saved to %s", log_file)


if __name__ == "__main__":
    main()
