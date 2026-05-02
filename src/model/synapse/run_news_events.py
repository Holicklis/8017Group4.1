from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from transformers import pipeline

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.append(str(_CURRENT_DIR))

from model import ETFNewsEngine, _default_paths

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configure_logging(level: int = logging.INFO) -> Path:
    log_dir = _project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "synapse_run_news_events.log"

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


def _load_financial_sentiment_model(model_name: str) -> Any:
    return pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
    )


def _normalize_sentiment_label(label: str) -> str:
    label_upper = label.upper()
    if "POS" in label_upper:
        return "positive"
    if "NEG" in label_upper:
        return "negative"
    return "neutral"


def _sentiment_to_score(label: str) -> int:
    if label == "positive":
        return 1
    if label == "negative":
        return -1
    return 0


def _score_with_sentiment(relevance_score: float, sentiment_score: int, confidence: float, alpha: float) -> float:
    return relevance_score * (1.0 + alpha * sentiment_score * confidence)


def _build_query_text(row: pd.Series, text_col: str) -> str:
    parts: List[str] = []
    headline = str(row.get(text_col, "")).strip()
    if headline:
        parts.append(headline)
    for col in ["Market_Event", "Sector", "Impact_Level", "Sentiment", "Market_Index"]:
        value = str(row.get(col, "")).strip()
        if value and value.lower() not in {"nan", "none"}:
            parts.append(value)
    return " | ".join(parts).strip()


def _prepare_output_dir(output_dir: Optional[Path]) -> Path:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    root = Path(__file__).resolve().parents[3] / "model_output" / "synpse"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = root / f"news_events_run_{ts}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_visuals(df_results: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    visuals: Dict[str, str] = {}
    top1_df = df_results[df_results["rank"] == 1].copy()

    ticker_counts = top1_df["predicted_ticker"].value_counts().head(12).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ticker_counts.index, ticker_counts.values)
    ax.set_title("Top Predicted Tickers (Top-1)")
    ax.set_xlabel("Matched News Count")
    ax.set_ylabel("Ticker")
    plt.tight_layout()
    png_path = output_dir / "top_ticker_frequency.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    visuals["top_ticker_frequency_png"] = str(png_path)

    if "Date" in top1_df.columns:
        day_df = top1_df.copy()
        day_df["Date"] = pd.to_datetime(day_df["Date"], errors="coerce")
        day_df = day_df.dropna(subset=["Date"])
        if not day_df.empty:
            daily = (
                day_df.groupby(day_df["Date"].dt.date)
                .agg(
                    news_count=("news_id", "count"),
                    avg_top1_score=("final_score", "mean"),
                )
                .reset_index()
            )
            daily["Date"] = daily["Date"].astype(str)
            line = px.line(
                daily,
                x="Date",
                y=["news_count", "avg_top1_score"],
                title="Daily News Volume and Average Top-1 Score",
                markers=True,
            )
            html_path = output_dir / "daily_score_volume.html"
            line.write_html(str(html_path), include_plotlyjs="cdn")
            visuals["daily_score_volume_html"] = str(html_path)

            if {"sentiment_adjusted_score", "sentiment_label"}.issubset(day_df.columns):
                daily_compare = (
                    day_df.groupby(day_df["Date"].dt.date)
                    .agg(
                        avg_relevance_score=("final_score", "mean"),
                        avg_combined_score=("sentiment_adjusted_score", "mean"),
                    )
                    .reset_index()
                )
                daily_compare["Date"] = daily_compare["Date"].astype(str)
                compare_fig = px.line(
                    daily_compare,
                    x="Date",
                    y=["avg_relevance_score", "avg_combined_score"],
                    title="Before vs After Sentiment Integration",
                    markers=True,
                )
                compare_path = output_dir / "before_after_sentiment_signal.html"
                compare_fig.write_html(str(compare_path), include_plotlyjs="cdn")
                visuals["before_after_sentiment_signal_html"] = str(compare_path)

    event_col = "Market_Event" if "Market_Event" in top1_df.columns else None
    if event_col:
        event_counts = top1_df.groupby([event_col, "predicted_ticker"]).size().reset_index(name="count")
        event_top = event_counts.sort_values("count", ascending=False).head(40)
        if not event_top.empty:
            bar = px.bar(
                event_top,
                x=event_col,
                y="count",
                color="predicted_ticker",
                title="Top Event-to-Ticker Match Patterns",
            )
            html_path = output_dir / "event_ticker_patterns.html"
            bar.write_html(str(html_path), include_plotlyjs="cdn")
            visuals["event_ticker_patterns_html"] = str(html_path)

    if {"sentiment_adjusted_score", "sentiment_label"}.issubset(top1_df.columns):
        sentiment_mix = top1_df["sentiment_label"].value_counts()
        sentiment_fig, sentiment_ax = plt.subplots(figsize=(8, 5))
        sentiment_ax.bar(sentiment_mix.index, sentiment_mix.values)
        sentiment_ax.set_title("Financial Sentiment Tag Distribution (Top-1)")
        sentiment_ax.set_xlabel("Sentiment")
        sentiment_ax.set_ylabel("News Count")
        plt.tight_layout()
        sentiment_path = output_dir / "sentiment_distribution.png"
        sentiment_fig.savefig(sentiment_path, dpi=150)
        plt.close(sentiment_fig)
        visuals["sentiment_distribution_png"] = str(sentiment_path)

    return visuals


def run_news_events(
    input_csv: Path,
    output_dir: Optional[Path],
    preset: str,
    corpus_mode: str,
    top_k: int,
    text_col: str,
    sentiment_model_name: str,
    sentiment_weight: float,
) -> Dict[str, object]:
    logger.info("Loading news events from %s", input_csv)
    paths = _default_paths()
    run_dir = _prepare_output_dir(output_dir)
    df_news = pd.read_csv(input_csv)
    if text_col not in df_news.columns:
        raise ValueError(f"Column '{text_col}' not found in {input_csv}")

    df_news = df_news.copy()
    df_news["query_text"] = df_news.apply(lambda row: _build_query_text(row, text_col=text_col), axis=1)
    df_news = df_news[df_news["query_text"].str.len() > 0].reset_index(drop=True)
    df_news["news_id"] = range(1, len(df_news) + 1)

    sentiment_input = (
        df_news[text_col].fillna("").astype(str).str.strip().where(lambda s: s.str.len() > 0, df_news["query_text"])
    )
    logger.info("Running sentiment model '%s' on %s records", sentiment_model_name, len(df_news))
    sentiment_model = _load_financial_sentiment_model(sentiment_model_name)
    sentiment_raw = sentiment_model(
        sentiment_input.tolist(),
        batch_size=32,
        truncation=True,
        max_length=256,
    )
    sentiment_labels = [_normalize_sentiment_label(str(row["label"])) for row in sentiment_raw]
    sentiment_conf = [float(row["score"]) for row in sentiment_raw]
    sentiment_numeric = [_sentiment_to_score(label) for label in sentiment_labels]
    df_news["sentiment_label"] = sentiment_labels
    df_news["sentiment_confidence"] = sentiment_conf
    df_news["sentiment_score"] = sentiment_numeric

    logger.info("Initializing ETFNewsEngine with preset=%s corpus_mode=%s", preset, corpus_mode)
    engine = ETFNewsEngine(
        **paths,
        preset=preset,
        corpus_mode=corpus_mode,
    )

    rows: List[Dict[str, object]] = []
    for _, row in tqdm(df_news.iterrows(), total=len(df_news), desc="Scoring news events"):
        query = str(row["query_text"])
        results = engine.search(query, top_k=top_k)
        sentiment_label = str(row["sentiment_label"])
        sentiment_confidence = float(row["sentiment_confidence"])
        sentiment_score = int(row["sentiment_score"])
        for rank, hit in enumerate(results, start=1):
            relevance_score = round(float(hit["final_score"]), 6)
            sentiment_adjusted = round(
                _score_with_sentiment(
                    relevance_score=relevance_score,
                    sentiment_score=sentiment_score,
                    confidence=sentiment_confidence,
                    alpha=sentiment_weight,
                ),
                6,
            )
            rows.append(
                {
                    "news_id": int(row["news_id"]),
                    "Date": row.get("Date"),
                    "Headline": row.get(text_col),
                    "Source": row.get("Source"),
                    "Market_Event": row.get("Market_Event"),
                    "Sector": row.get("Sector"),
                    "query_text": query,
                    "rank": rank,
                    "predicted_ticker": hit["ticker"],
                    "final_score": relevance_score,
                    "bi_score": round(float(hit["bi_score"]), 6),
                    "cross_score": round(float(hit["cross_score"]), 6),
                    "boost": round(float(hit["boost"]), 6),
                    "sentiment_label": sentiment_label,
                    "sentiment_confidence": round(sentiment_confidence, 6),
                    "sentiment_score": sentiment_score,
                    "sentiment_adjusted_score": sentiment_adjusted,
                }
            )

    df_results = pd.DataFrame(rows)
    topk_path = run_dir / "news_event_topk_matches.csv"
    df_results.to_csv(topk_path, index=False, encoding="utf-8-sig")

    top1 = df_results[df_results["rank"] == 1].copy()
    top1_path = run_dir / "news_event_top1_summary.csv"
    top1.to_csv(top1_path, index=False, encoding="utf-8-sig")

    visuals = _save_visuals(df_results, run_dir)
    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(run_dir),
        "rows_input": int(len(df_news)),
        "rows_output_topk": int(len(df_results)),
        "rows_output_top1": int(len(top1)),
        "preset": preset,
        "corpus_mode": corpus_mode,
        "top_k": top_k,
        "text_col": text_col,
        "sentiment_model": sentiment_model_name,
        "sentiment_weight": sentiment_weight,
        "topk_output_csv": str(topk_path),
        "top1_output_csv": str(top1_path),
        "visualizations": visuals,
        "top_ticker_counts": top1["predicted_ticker"].value_counts().head(15).to_dict(),
        "sentiment_counts": top1["sentiment_label"].value_counts().to_dict(),
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    logger.info("News event run completed. Summary saved to %s", summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Synapse on a financial news CSV and export visuals")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preset", type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument("--corpus-mode", type=str, default="profile", choices=["profile", "sentence"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--text-col", type=str, default="Headline")
    parser.add_argument("--sentiment-model", type=str, default="ProsusAI/finbert")
    parser.add_argument(
        "--sentiment-weight",
        type=float,
        default=0.25,
        help="Weight for sentiment-adjusted relevance score",
    )
    return parser.parse_args()


def main() -> None:
    log_file = configure_logging()
    args = _parse_args()
    result = run_news_events(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        preset=args.preset,
        corpus_mode=args.corpus_mode,
        top_k=max(args.top_k, 1),
        text_col=args.text_col,
        sentiment_model_name=args.sentiment_model,
        sentiment_weight=max(args.sentiment_weight, 0.0),
    )
    logger.info("Run completed. summary_json=%s", result.get("summary_json"))
    logger.debug("Run payload: %s", json.dumps(result, indent=2))
    logger.info("Logs saved to %s", log_file)


if __name__ == "__main__":
    main()
