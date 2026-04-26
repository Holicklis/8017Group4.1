from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from synthesis_engine import DEFAULT_LOCAL_QWEN_MODEL, generate_synthesis


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    return _project_root() / "model_output" / "Synthesis"


def run_synthesis_pipeline(
    output_root: Path | None = None,
    ticker: str | None = None,
    query: str | None = None,
    backend: str = "transformers",
    model_name: str = DEFAULT_LOCAL_QWEN_MODEL,
    enable_query_similarity: bool = False,
    enable_response_cache: bool = True,
) -> dict[str, str]:
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


if __name__ == "__main__":
    args = _parse_args()
    print(
        json.dumps(
            run_synthesis_pipeline(
                output_root=args.output_root,
                ticker=args.ticker,
                query=args.query,
                backend=args.backend,
                model_name=args.model,
                enable_query_similarity=args.enable_query_similarity,
                enable_response_cache=not args.disable_response_cache,
            ),
            indent=2,
        )
    )
