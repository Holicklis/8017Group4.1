from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.append(str(_CURRENT_DIR))

from model import run_side_by_side_evaluation

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configure_logging(level: int = logging.INFO) -> Path:
    log_dir = _project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "synapse_evaluate_benchmark.log"

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


def _default_output_path() -> Path:
    output_dir = Path(__file__).resolve().parents[3] / "model_output" / "synpse"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"benchmark_side_by_side_{timestamp}.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Synapse side-by-side benchmark")
    parser.add_argument("--preset", type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--sentence-row-cap",
        type=int,
        default=4000,
        help="Cap legacy sentence corpus rows for evaluation runtime",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output json path")
    return parser.parse_args()


def main() -> None:
    log_file = configure_logging()
    args = _parse_args()
    logger.info("Running Synapse benchmark with preset=%s, top_k=%s", args.preset, args.top_k)
    results = run_side_by_side_evaluation(
        preset=args.preset,
        top_k=max(args.top_k, 3),
        sentence_row_cap=max(args.sentence_row_cap, 500),
    )
    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Benchmark results saved to %s", output_path)
    logger.debug("Benchmark payload: %s", json.dumps(results, indent=2))
    logger.info("Logs saved to %s", log_file)


if __name__ == "__main__":
    main()
