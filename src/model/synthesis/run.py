from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from synthesis_engine import DEFAULT_LOCAL_QWEN_MODEL, generate_synthesis

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    return _project_root() / "model_output" / "Synthesis"


def configure_logging(level: int = logging.INFO) -> Path:
    log_dir = _project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "synthesis_run.log"

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


def run_synthesis_pipeline(
    output_root: Path | None = None,
    ticker: str | None = None,
    query: str | None = None,
    backend: str = "transformers",
    model_name: str = DEFAULT_LOCAL_QWEN_MODEL,
    enable_query_similarity: bool = False,
    enable_response_cache: bool = True,
) -> dict[str, object]:
    logger.info("Starting synthesis pipeline")
    root = output_root or _default_output_root()
    root.mkdir(parents=True, exist_ok=True)

    if not ticker or not query:
        summary = {
            "output_root": str(root),
            "status": "ready",
            "message": ("Synthesis engine is available. Provide --ticker and --query to run local Qwen inference."),
            "timestamp": datetime.now().isoformat(),
        }
    else:
        logger.info("Generating synthesis for ticker=%s with backend=%s", ticker, backend)
        result = generate_synthesis(
            ticker=ticker,
            user_query=query,
            backend=backend,
            qwen_model=model_name,
            enable_query_similarity=enable_query_similarity,
            enable_response_cache=enable_response_cache,
        )
        summary = {
            "output_root": str(root),
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "query": query,
            "backend": backend,
            "model": model_name,
            "enable_query_similarity": enable_query_similarity,
            "enable_response_cache": enable_response_cache,
            "result": result,
        }

    summary_path = root / "synthesis_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    logger.info("Synthesis run completed. Summary saved to %s", summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Financial Synthesis model pipeline.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--ticker", type=str, default=None, help="Target HK ETF ticker, e.g. 2800")
    parser.add_argument("--query", type=str, default=None, help="User question for synthesis advisor")
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["ollama", "vllm", "transformers"],
        help="Inference backend (default transformers for in-process local model)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LOCAL_QWEN_MODEL,
        help="Model identifier (HF path for transformers, or tag/id for ollama/vllm)",
    )
    parser.add_argument(
        "--enable-query-similarity",
        action="store_true",
        help="Enable extra sentence-embedding rerank against query (slower).",
    )
    parser.add_argument(
        "--disable-response-cache",
        action="store_true",
        help="Disable synthesis response cache and force fresh generation.",
    )
    return parser.parse_args()


def main() -> None:
    log_file = configure_logging()
    args = _parse_args()
    result = run_synthesis_pipeline(
        output_root=args.output_root,
        ticker=args.ticker,
        query=args.query,
        backend=args.backend,
        model_name=args.model,
        enable_query_similarity=args.enable_query_similarity,
        enable_response_cache=not args.disable_response_cache,
    )
    logger.info("Run completed. summary_json=%s", result.get("summary_json"))
    logger.debug("Run payload: %s", json.dumps(result, indent=2))
    logger.info("Logs saved to %s", log_file)


if __name__ == "__main__":
    main()
