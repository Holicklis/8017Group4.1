from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from tqdm import tqdm

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.append(str(_CURRENT_DIR))

from model import ETFNewsEngine, _default_paths


@dataclass(frozen=True)
class TopicTemplate:
    domain: str
    asset_phrase: str
    catalysts: Sequence[str]
    risk_drivers: Sequence[str]


TOPIC_LIBRARY: Sequence[TopicTemplate] = [
    TopicTemplate(
        "equity",
        "regional equity baskets",
        ["earnings season update", "corporate guidance release", "equity outlook briefing"],
        ["earnings downgrade", "revenue slowdown", "valuation compression"],
    ),
    TopicTemplate(
        "rates",
        "duration-sensitive bond exposures",
        ["FOMC meeting", "central bank policy statement", "inflation print release"],
        ["yield curve steepening", "rate hike expectations", "term premium shock"],
    ),
    TopicTemplate(
        "commodity",
        "commodity-linked holdings",
        ["OPEC supply announcement", "inventory report update", "commodity market briefing"],
        ["energy spike", "inventory drawdown", "raw material shortage"],
    ),
    TopicTemplate(
        "crypto",
        "crypto and digital-asset proxies",
        ["crypto regulation hearing", "exchange policy update", "risk sentiment shift"],
        ["bitcoin volatility", "risk-off flows", "regulatory uncertainty"],
    ),
    TopicTemplate(
        "fx",
        "currency-sensitive allocations",
        ["currency policy signal", "FX reserve update", "cross-border flow data"],
        ["usd strength", "rmb weakness", "hedging cost increase"],
    ),
    TopicTemplate(
        "credit",
        "corporate credit exposures",
        ["credit market update", "funding conditions release", "default cycle signal"],
        ["spread widening", "refinancing pressure", "default risk repricing"],
    ),
    TopicTemplate(
        "china",
        "mainland and china policy trades",
        ["China policy meeting", "mainland macro briefing", "regulatory circular"],
        ["policy easing", "property stress", "regulatory recalibration"],
    ),
    TopicTemplate(
        "us_macro",
        "us macro-sensitive allocations",
        ["US labor report", "Fed communication", "US fiscal update"],
        ["fed pivot risk", "labor data miss", "fiscal uncertainty"],
    ),
    TopicTemplate(
        "tech",
        "technology and innovation themes",
        ["big tech guidance release", "AI investment update", "semiconductor cycle briefing"],
        ["ai cycle repricing", "semiconductor weakness", "software demand slowdown"],
    ),
    TopicTemplate(
        "money_market",
        "cash and money-market sleeves",
        ["short-rate policy update", "funding market signal", "liquidity operations notice"],
        ["short-rate repricing", "funding pressure", "liquidity preference"],
    ),
]

PARAPHRASE_TEMPLATES = [
    "{catalyst} is signaling {driver} for {asset}.",
    "{asset} are reacting as {catalyst} points to {driver}.",
    "{catalyst} now suggests {driver}, putting pressure on {asset}.",
    "{asset} are being repriced because {catalyst} indicates {driver}.",
    "{driver} is being signaled by {catalyst}, affecting {asset}.",
    "After {catalyst}, markets are pricing in {driver} for {asset}.",
    "{catalyst} implies {driver}, so {asset} are under review.",
    "Investors read {catalyst} as a sign of {driver} for {asset}.",
    "{asset} face renewed risk after {catalyst} signaled {driver}.",
    "{driver} expectations increased following {catalyst}, impacting {asset}.",
    "{catalyst} has shifted expectations toward {driver} for {asset}.",
    "With {catalyst} released, {asset} are exposed to {driver}.",
    "{asset} are sensitive to {driver}, and {catalyst} reinforces that view.",
    "{catalyst} reinforces {driver} concerns around {asset}.",
    "{driver} risk is rising for {asset} as shown by {catalyst}.",
    "The message from {catalyst} is {driver}, which affects {asset}.",
    "{asset} are being reassessed since {catalyst} indicates {driver}.",
    "Following {catalyst}, traders link {asset} to {driver}.",
    "{catalyst} and {driver} together are driving repricing in {asset}.",
    "{asset} pricing now reflects {driver} signaled by {catalyst}.",
]


def _prepare_output_dir(base_dir: Path | None) -> Path:
    if base_dir:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    root = Path(__file__).resolve().parents[3] / "model_output" / "synpse"
    run_dir = root / f"stability_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _compose_paraphrase(variant_id: int, catalyst: str, asset: str, driver: str) -> str:
    template = PARAPHRASE_TEMPLATES[(variant_id - 1) % len(PARAPHRASE_TEMPLATES)]
    return template.format(catalyst=catalyst, asset=asset, driver=driver)


def generate_synthetic_news(num_concepts: int = 100, variants_per_concept: int = 20) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for concept_id in range(1, num_concepts + 1):
        topic = TOPIC_LIBRARY[(concept_id - 1) % len(TOPIC_LIBRARY)]
        catalyst = topic.catalysts[(concept_id - 1) % len(topic.catalysts)]
        driver = topic.risk_drivers[(concept_id - 1) % len(topic.risk_drivers)]
        concept_meaning = f"{catalyst} signals {driver} for {topic.asset_phrase}"
        for variant_id in range(1, variants_per_concept + 1):
            headline = _compose_paraphrase(
                variant_id=variant_id,
                catalyst=catalyst,
                asset=topic.asset_phrase,
                driver=driver,
            )
            rows.append(
                {
                    "concept_id": concept_id,
                    "domain": topic.domain,
                    "variant_id": variant_id,
                    "base_meaning": concept_meaning,
                    "headline": headline,
                }
            )
    return pd.DataFrame(rows)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def _safe_corr(values_a: Sequence[float], values_b: Sequence[float], method: str = "pearson") -> float:
    series_a = pd.Series(values_a, dtype="float64")
    series_b = pd.Series(values_b, dtype="float64")
    corr = series_a.corr(series_b, method=method)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _finite_mean(values: Sequence[float]) -> float:
    cleaned = [float(value) for value in values if pd.notna(value) and math.isfinite(float(value))]
    if not cleaned:
        return 0.0
    return float(sum(cleaned) / len(cleaned))


def _evaluate_stability(df_predictions: pd.DataFrame, top_k: int, tie_epsilon: float = 0.01) -> Dict[str, pd.DataFrame]:
    concept_rows: List[Dict[str, object]] = []
    for concept_id, concept_group in df_predictions.groupby("concept_id"):
        topk_lists: List[List[str]] = []
        topk_score_lists: List[List[float]] = []
        topk_score_maps: List[Dict[str, float]] = []
        top1: List[str] = []
        score_spreads: List[float] = []
        tie_buckets: List[set[str]] = []
        domain = str(concept_group["domain"].iloc[0])
        for _, sample in concept_group.groupby("variant_id"):
            ranked = sample.sort_values("rank")
            tickers = ranked["predicted_ticker"].astype(str).tolist()[:top_k]
            scores = ranked["final_score"].astype(float).tolist()[:top_k]
            if not tickers:
                continue
            min_length = min(len(tickers), len(scores))
            tickers = tickers[:min_length]
            scores = scores[:min_length]
            topk_lists.append(tickers)
            topk_score_lists.append(scores)
            topk_score_maps.append(dict(zip(tickers, scores)))
            top1.append(tickers[0])
            top_score = scores[0]
            tie_bucket = {ticker for ticker, score in zip(tickers, scores) if (top_score - score) <= tie_epsilon}
            tie_buckets.append(tie_bucket)
            if len(scores) >= 2:
                score_spreads.append(float(scores[0] - scores[-1]))
        if len(topk_lists) < 2:
            continue

        floor_score = min(score for score_list in topk_score_lists for score in score_list) - 0.05
        pairwise_jaccard: List[float] = []
        pairwise_score_corr_pearson: List[float] = []
        pairwise_score_corr_spearman: List[float] = []
        pairwise_rank_score_corr: List[float] = []
        pairwise_gap_delta: List[float] = []
        pairwise_tie_bucket_jaccard: List[float] = []

        for i, j in combinations(range(len(topk_lists)), 2):
            tickers_i, tickers_j = topk_lists[i], topk_lists[j]
            map_i, map_j = topk_score_maps[i], topk_score_maps[j]
            score_i, score_j = topk_score_lists[i], topk_score_lists[j]

            pairwise_jaccard.append(_jaccard(tickers_i, tickers_j))

            union_tickers = sorted(set(tickers_i).union(set(tickers_j)))
            vec_i = [map_i.get(ticker, floor_score) for ticker in union_tickers]
            vec_j = [map_j.get(ticker, floor_score) for ticker in union_tickers]
            pairwise_score_corr_pearson.append(_safe_corr(vec_i, vec_j, method="pearson"))
            pairwise_score_corr_spearman.append(_safe_corr(vec_i, vec_j, method="spearman"))

            length = min(len(score_i), len(score_j))
            pairwise_rank_score_corr.append(_safe_corr(score_i[:length], score_j[:length], method="pearson"))
            pairwise_gap_delta.append(abs((score_i[0] - score_i[-1]) - (score_j[0] - score_j[-1])))
            pairwise_tie_bucket_jaccard.append(_jaccard(list(tie_buckets[i]), list(tie_buckets[j])))

        top1_consistency = pd.Series(top1).value_counts(normalize=True).iloc[0]
        concept_rows.append(
            {
                "concept_id": int(concept_id),
                "domain": domain,
                "avg_topk_jaccard": round(_finite_mean(pairwise_jaccard), 4),
                "top1_consistency": round(float(top1_consistency), 4),
                "avg_score_corr_ticker_pearson": round(_finite_mean(pairwise_score_corr_pearson), 4),
                "avg_score_corr_ticker_spearman": round(_finite_mean(pairwise_score_corr_spearman), 4),
                "avg_rank_score_corr": round(_finite_mean(pairwise_rank_score_corr), 4),
                "avg_score_gap_delta": round(_finite_mean(pairwise_gap_delta), 4),
                "avg_score_spread": round(_finite_mean(score_spreads), 4) if score_spreads else 0.0,
                "avg_tie_bucket_jaccard": round(_finite_mean(pairwise_tie_bucket_jaccard), 4),
                "variants": int(len(topk_lists)),
            }
        )

    df_concept = pd.DataFrame(concept_rows).sort_values("concept_id").reset_index(drop=True)
    df_domain = (
        df_concept.groupby("domain")
        .agg(
            concepts=("concept_id", "count"),
            avg_topk_jaccard=("avg_topk_jaccard", "mean"),
            avg_top1_consistency=("top1_consistency", "mean"),
            avg_score_corr_ticker_pearson=("avg_score_corr_ticker_pearson", "mean"),
            avg_score_corr_ticker_spearman=("avg_score_corr_ticker_spearman", "mean"),
            avg_rank_score_corr=("avg_rank_score_corr", "mean"),
            avg_score_gap_delta=("avg_score_gap_delta", "mean"),
            avg_score_spread=("avg_score_spread", "mean"),
            avg_tie_bucket_jaccard=("avg_tie_bucket_jaccard", "mean"),
        )
        .reset_index()
    )
    df_domain["avg_topk_jaccard"] = df_domain["avg_topk_jaccard"].round(4)
    df_domain["avg_top1_consistency"] = df_domain["avg_top1_consistency"].round(4)
    df_domain["avg_score_corr_ticker_pearson"] = df_domain["avg_score_corr_ticker_pearson"].round(4)
    df_domain["avg_score_corr_ticker_spearman"] = df_domain["avg_score_corr_ticker_spearman"].round(4)
    df_domain["avg_rank_score_corr"] = df_domain["avg_rank_score_corr"].round(4)
    df_domain["avg_score_gap_delta"] = df_domain["avg_score_gap_delta"].round(4)
    df_domain["avg_score_spread"] = df_domain["avg_score_spread"].round(4)
    df_domain["avg_tie_bucket_jaccard"] = df_domain["avg_tie_bucket_jaccard"].round(4)
    return {"concept": df_concept, "domain": df_domain}


def _save_plots(df_concept: pd.DataFrame, df_domain: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    outputs: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_concept["avg_topk_jaccard"], bins=20)
    ax.set_title("Distribution of Top-5 Stability (Jaccard)")
    ax.set_xlabel("Average pairwise top-5 Jaccard")
    ax.set_ylabel("Concept count")
    plt.tight_layout()
    hist_path = output_dir / "stability_jaccard_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    outputs["stability_jaccard_histogram_png"] = str(hist_path)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ordered = df_domain.sort_values("avg_topk_jaccard", ascending=True)
    ax2.barh(ordered["domain"], ordered["avg_topk_jaccard"])
    ax2.set_title("Average Stability by Domain")
    ax2.set_xlabel("Average Top-5 Jaccard")
    ax2.set_ylabel("Domain")
    plt.tight_layout()
    domain_path = output_dir / "stability_by_domain.png"
    fig2.savefig(domain_path, dpi=150)
    plt.close(fig2)
    outputs["stability_by_domain_png"] = str(domain_path)

    scatter = px.scatter(
        df_concept,
        x="avg_score_corr_ticker_spearman",
        y="top1_consistency",
        color="domain",
        title="Concept-Level Stability Map (Score Correlation vs Top-1 Consistency)",
        hover_data=["concept_id", "variants", "avg_topk_jaccard", "avg_rank_score_corr", "avg_score_gap_delta"],
    )
    scatter_path = output_dir / "concept_stability_map.html"
    scatter.write_html(str(scatter_path), include_plotlyjs="cdn")
    outputs["concept_stability_map_html"] = str(scatter_path)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(df_concept["avg_score_corr_ticker_spearman"], bins=20)
    ax3.set_title("Distribution of Ticker-Aligned Score Correlation (Spearman)")
    ax3.set_xlabel("Average pairwise score correlation")
    ax3.set_ylabel("Concept count")
    plt.tight_layout()
    corr_hist_path = output_dir / "stability_score_correlation_histogram.png"
    fig3.savefig(corr_hist_path, dpi=150)
    plt.close(fig3)
    outputs["stability_score_correlation_histogram_png"] = str(corr_hist_path)
    return outputs


def run_stability_assessment(
    output_dir: Path | None,
    preset: str,
    corpus_mode: str,
    top_k: int,
    num_concepts: int,
    variants_per_concept: int,
    tie_epsilon: float,
    use_light_rerank: bool,
    cross_top_n: int,
    apply_query_canonicalization: bool,
) -> Dict[str, object]:
    out_dir = _prepare_output_dir(output_dir)
    synthetic = generate_synthetic_news(num_concepts=num_concepts, variants_per_concept=variants_per_concept)
    synthetic_path = out_dir / "synthetic_financial_headlines.csv"
    synthetic.to_csv(synthetic_path, index=False, encoding="utf-8-sig")

    engine = ETFNewsEngine(
        **_default_paths(),
        preset=preset,
        corpus_mode=corpus_mode,
        use_cross_encoder=use_light_rerank,
        cross_encoder_top_n=cross_top_n if use_light_rerank else 0,
        apply_query_canonicalization=apply_query_canonicalization,
    )

    prediction_rows: List[Dict[str, object]] = []
    for _, row in tqdm(synthetic.iterrows(), total=len(synthetic), desc="Running stability retrieval"):
        hits = engine.search(str(row["headline"]), top_k=top_k)
        for rank, hit in enumerate(hits, start=1):
            prediction_rows.append(
                {
                    "concept_id": int(row["concept_id"]),
                    "domain": str(row["domain"]),
                    "variant_id": int(row["variant_id"]),
                    "headline": str(row["headline"]),
                    "rank": rank,
                    "predicted_ticker": str(hit["ticker"]).zfill(5),
                    "final_score": round(float(hit["final_score"]), 6),
                }
            )

    df_predictions = pd.DataFrame(prediction_rows)
    prediction_path = out_dir / "stability_topk_predictions.csv"
    df_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    stability = _evaluate_stability(df_predictions, top_k=top_k, tie_epsilon=tie_epsilon)
    df_concept = stability["concept"]
    df_domain = stability["domain"]
    concept_path = out_dir / "concept_stability.csv"
    domain_path = out_dir / "domain_stability.csv"
    df_concept.to_csv(concept_path, index=False, encoding="utf-8-sig")
    df_domain.to_csv(domain_path, index=False, encoding="utf-8-sig")

    plots = _save_plots(df_concept, df_domain, out_dir)
    summary = {
        "output_dir": str(out_dir),
        "preset": preset,
        "corpus_mode": corpus_mode,
        "top_k": top_k,
        "num_concepts": num_concepts,
        "variants_per_concept": variants_per_concept,
        "tie_epsilon": tie_epsilon,
        "use_light_rerank": use_light_rerank,
        "cross_top_n": cross_top_n if use_light_rerank else 0,
        "apply_query_canonicalization": apply_query_canonicalization,
        "synthetic_rows": int(len(synthetic)),
        "prediction_rows": int(len(df_predictions)),
        "overall_avg_topk_jaccard": round(float(df_concept["avg_topk_jaccard"].mean()), 4),
        "overall_avg_top1_consistency": round(float(df_concept["top1_consistency"].mean()), 4),
        "overall_avg_tie_bucket_jaccard": round(float(df_concept["avg_tie_bucket_jaccard"].mean()), 4),
        "overall_avg_score_corr_ticker_pearson": round(float(df_concept["avg_score_corr_ticker_pearson"].mean()), 4),
        "overall_avg_score_corr_ticker_spearman": round(float(df_concept["avg_score_corr_ticker_spearman"].mean()), 4),
        "overall_avg_rank_score_corr": round(float(df_concept["avg_rank_score_corr"].mean()), 4),
        "overall_avg_score_gap_delta": round(float(df_concept["avg_score_gap_delta"].mean()), 4),
        "overall_avg_score_spread": round(float(df_concept["avg_score_spread"].mean()), 4),
        "synthetic_csv": str(synthetic_path),
        "predictions_csv": str(prediction_path),
        "concept_stability_csv": str(concept_path),
        "domain_stability_csv": str(domain_path),
        "plots": plots,
    }
    summary_path = out_dir / "stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic stability test for Synapse top-k retrieval")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preset", type=str, default="fast", choices=["fast", "quality"])
    parser.add_argument("--corpus-mode", type=str, default="profile", choices=["profile", "sentence"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-concepts", type=int, default=100)
    parser.add_argument("--variants-per-concept", type=int, default=20)
    parser.add_argument("--tie-epsilon", type=float, default=0.01)
    parser.add_argument("--enable-light-rerank", action="store_true")
    parser.add_argument("--cross-top-n", type=int, default=12)
    parser.add_argument("--disable-query-canonicalization", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_stability_assessment(
        output_dir=args.output_dir,
        preset=args.preset,
        corpus_mode=args.corpus_mode,
        top_k=max(args.top_k, 2),
        num_concepts=max(args.num_concepts, 10),
        variants_per_concept=max(args.variants_per_concept, 5),
        tie_epsilon=max(args.tie_epsilon, 0.0),
        use_light_rerank=args.enable_light_rerank,
        cross_top_n=max(args.cross_top_n, 2),
        apply_query_canonicalization=not args.disable_query_canonicalization,
    )
    print(json.dumps(result, indent=2))
