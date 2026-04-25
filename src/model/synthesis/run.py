from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    return _project_root() / "model_output" / "Synthesis"


def run_synthesis_pipeline(output_root: Path | None = None) -> dict[str, str]:
    root = output_root or _default_output_root()
    root.mkdir(parents=True, exist_ok=True)

    # Placeholder until synthesis engine is fully implemented.
    summary = {
        "output_root": str(root),
        "status": "placeholder",
        "message": (
            "Financial Synthesis run.py is ready as a unified entrypoint. "
            "Integrate synthesis inference modules here when implementation is available."
        ),
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = root / "synthesis_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Financial Synthesis model pipeline.")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(json.dumps(run_synthesis_pipeline(output_root=args.output_root), indent=2))
