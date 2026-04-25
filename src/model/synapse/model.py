from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from tqdm import tqdm

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))

from text_extraction.pdf_text_extractor import ETFPDFProcessor


@dataclass(frozen=True)
class ModelPreset:
    name: str
    bi_encoder: str
    cross_encoder: str
    use_cross_encoder: bool
    cross_encoder_top_n: int
    retrieval_multiplier: int
    cross_encoder_weight: float
    ticker_weight: float
    metadata_weight: float
    tag_weight: float


MODEL_PRESETS: Dict[str, ModelPreset] = {
    "fast": ModelPreset(
        name="fast",
        bi_encoder="intfloat/e5-small-v2",
        cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder=False,
        cross_encoder_top_n=0,
        retrieval_multiplier=4,
        cross_encoder_weight=0.12,
        ticker_weight=0.35,
        metadata_weight=0.10,
        tag_weight=0.10,
    ),
    "quality": ModelPreset(
        name="quality",
        bi_encoder="BAAI/bge-m3",
        cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder=True,
        cross_encoder_top_n=24,
        retrieval_multiplier=6,
        cross_encoder_weight=0.20,
        ticker_weight=0.35,
        metadata_weight=0.12,
        tag_weight=0.12,
    ),
}


class ETFNewsEngine:
    def __init__(
        self,
        documentation_dir: str,
        metadata_excel: str,
        profile_csv: Optional[str] = None,
        cache_dir: Optional[str] = None,
        preset: str = "fast",
        corpus_mode: str = "profile",
        use_cross_encoder: Optional[bool] = None,
        cross_encoder_top_n: Optional[int] = None,
        sentence_row_cap: Optional[int] = None,
        apply_query_canonicalization: bool = True,
    ):
        if preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {', '.join(MODEL_PRESETS)}")
        if corpus_mode not in {"profile", "sentence"}:
            raise ValueError("corpus_mode must be 'profile' or 'sentence'")

        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.preset = MODEL_PRESETS[preset]
        self.corpus_mode = corpus_mode
        self.use_cross_encoder = self.preset.use_cross_encoder if use_cross_encoder is None else use_cross_encoder
        self.cross_encoder_top_n = (
            self.preset.cross_encoder_top_n if cross_encoder_top_n is None else max(cross_encoder_top_n, 0)
        )
        self.documentation_dir = Path(documentation_dir)
        self.metadata_excel = Path(metadata_excel)
        self.profile_csv = Path(profile_csv) if profile_csv else self._default_profile_path()
        self.cache_dir = Path(cache_dir) if cache_dir else self._default_cache_dir()
        self.sentence_row_cap = sentence_row_cap
        self.apply_query_canonicalization = apply_query_canonicalization
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing Synapse engine on: {self.device} ({self.preset.name} preset, {self.corpus_mode} corpus)")
        self.metadata_df = self._load_metadata(self.metadata_excel)
        self._prepare_metadata_cache()
        self.docs_df = self._load_corpus()
        self.profile_version = self._resolve_profile_version()

        print("Loading embedding model...")
        self.bi_encoder = SentenceTransformer(self.preset.bi_encoder, device=self.device)
        self.cross_encoder = None
        if self.use_cross_encoder and self.cross_encoder_top_n > 0:
            print("Loading cross-encoder model...")
            self.cross_encoder = CrossEncoder(self.preset.cross_encoder, device=self.device)

        self.corpus_embeddings = self._load_or_build_embeddings()

    def _default_profile_path(self) -> Path:
        project_root = self.documentation_dir.parents[2]
        return project_root / "model_output" / "synpse" / "etf_profiles.csv"

    def _default_cache_dir(self) -> Path:
        project_root = self.documentation_dir.parents[2]
        return project_root / "model_output" / "synpse" / "cache"

    def _load_metadata(self, metadata_excel: Path) -> pd.DataFrame:
        metadata = pd.read_excel(metadata_excel).head(-2)
        metadata["ticker"] = (
            pd.to_numeric(metadata["Stock code*"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
        return metadata

    def _prepare_metadata_cache(self) -> None:
        self.meta_lookup: Dict[str, Dict[str, str]] = {}
        for _, row in self.metadata_df.iterrows():
            ticker = row["ticker"]
            record = row.to_dict()
            self.meta_lookup[ticker] = record
            self.meta_lookup[str(int(ticker))] = record

    def _load_corpus(self) -> pd.DataFrame:
        if self.corpus_mode == "profile":
            return self._load_profile_corpus()
        return self._load_sentence_corpus()

    def _load_profile_corpus(self) -> pd.DataFrame:
        if not self.profile_csv.exists():
            print(f"Profile corpus not found at {self.profile_csv}; generating profiles now...")
            processor = ETFPDFProcessor(data_root=self.documentation_dir.parents[1])
            generated = processor.generate_etf_profiles(output_path=self.profile_csv)
            if not generated:
                raise FileNotFoundError(f"Unable to generate profile corpus at {self.profile_csv}")

        df = pd.read_csv(self.profile_csv)
        required_columns = {"ticker", "profile_text"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Profile csv missing required columns: {sorted(missing)}")

        df = df.copy()
        df["ticker"] = df["ticker"].astype(str).str.zfill(5)
        df["text"] = df["profile_text"].astype(str).str.strip()
        df = df[df["text"] != ""].reset_index(drop=True)
        return df

    def _load_sentence_corpus(self) -> pd.DataFrame:
        all_dfs: List[pd.DataFrame] = []
        ticker_dirs = [path for path in self.documentation_dir.iterdir() if path.is_dir()]
        print(f"Found {len(ticker_dirs)} ETF folders. Reading sentence corpus...")
        for ticker_dir in tqdm(ticker_dirs, desc="Reading sentence CSVs"):
            ticker = ticker_dir.name
            for csv_file in sorted((ticker_dir / "csv").glob("*.csv")):
                file_name = csv_file.name.lower()
                if "prospectus" not in file_name and "key_facts" not in file_name and "key facts" not in file_name:
                    continue
                try:
                    temp_df = pd.read_csv(csv_file, usecols=["text"])
                    temp_df = temp_df.dropna().copy()
                    temp_df["text"] = temp_df["text"].astype(str).str.strip()
                    temp_df = temp_df[temp_df["text"].str.len() > 20]
                    if temp_df.empty:
                        continue
                    temp_df["ticker"] = ticker
                    all_dfs.append(temp_df[["ticker", "text"]])
                except Exception:
                    continue
        if not all_dfs:
            raise ValueError("No sentence-level corpus rows found in documentation directory.")
        sentence_df = pd.concat(all_dfs, ignore_index=True).reset_index(drop=True)
        if self.sentence_row_cap and self.sentence_row_cap > 0 and len(sentence_df) > self.sentence_row_cap:
            ticker_count = max(sentence_df["ticker"].nunique(), 1)
            per_ticker_cap = max(1, self.sentence_row_cap // ticker_count)
            sentence_df = (
                sentence_df.groupby("ticker", group_keys=False)
                .head(per_ticker_cap)
                .reset_index(drop=True)
                .head(self.sentence_row_cap)
            )
            print(
                f"Sentence corpus capped to {len(sentence_df)} rows "
                f"(requested cap={self.sentence_row_cap}, per ticker={per_ticker_cap})"
            )
        return sentence_df

    def _resolve_profile_version(self) -> str:
        if "profile_version" in self.docs_df.columns and not self.docs_df["profile_version"].isna().all():
            return str(self.docs_df["profile_version"].fillna("unknown").mode().iloc[0])
        return "legacy_sentence_corpus"

    def _cache_key(self) -> str:
        payload = {
            "preset": self.preset.name,
            "bi_encoder": self.preset.bi_encoder,
            "corpus_mode": self.corpus_mode,
            "profile_version": self.profile_version,
            "rows": int(len(self.docs_df)),
            "text_char_sum": int(self.docs_df["text"].str.len().sum()),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _load_or_build_embeddings(self) -> torch.Tensor:
        cache_file = self.cache_dir / f"corpus_embeddings_{self._cache_key()}.pt"
        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}")
            return torch.load(cache_file, map_location=self.device)

        print(f"Encoding {len(self.docs_df)} corpus rows with {self.preset.bi_encoder}...")
        embeddings = self.bi_encoder.encode(
            self.docs_df["text"].tolist(),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32 if self.preset.name == "fast" else 16,
        )
        torch.save(embeddings, cache_file)
        print(f"Saved embeddings cache to {cache_file}")
        return embeddings

    def _iter_metadata_fields(self, ticker: str) -> Iterable[str]:
        meta = self.meta_lookup.get(ticker)
        if not meta:
            return []
        fields = [
            "Asset class*",
            "Investment focus*",
            "Country of domicile*",
            "Geographic focus*",
            "Benchmark*",
        ]
        values = [str(meta.get(field, "")).lower().strip() for field in fields]
        return [value for value in values if value and value != "nan"]

    def _calculate_boost(self, news_text: str, row: pd.Series) -> float:
        news_lower = news_text.lower()
        ticker = str(row["ticker"])
        boost = 0.0

        if ticker in news_lower or str(int(ticker)) in news_lower:
            boost += self.preset.ticker_weight

        for value in self._iter_metadata_fields(ticker):
            if len(value) >= 4 and value in news_lower:
                boost += self.preset.metadata_weight

        for column in ["risk_tags", "component_tags"]:
            if column in row and pd.notna(row[column]):
                for tag in str(row[column]).split(","):
                    normalized = tag.strip().replace("_", " ")
                    if normalized and normalized in news_lower:
                        boost += self.preset.tag_weight
        return boost

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _canonicalize_query(news_text: str) -> str:
        normalized = news_text.lower()
        substitution_rules = [
            (r"\bfomc\b|\bfed\b|\bfederal reserve\b", "fomc"),
            (r"\brate hikes?\b|\bhigher rates?\b|\bmonetary tightening\b", "interest rate hike"),
            (r"\brate cuts?\b|\beasing cycle\b", "interest rate cut"),
            (r"\bnonfarm payrolls?\b|\bus labor report\b", "us labor data"),
            (r"\bhang seng tech\b|\bhk tech\b", "hong kong technology equity"),
            (r"\bbitcoin\b|\bbtc\b", "bitcoin"),
            (r"\boil\b|\bcrude\b", "oil commodity"),
            (r"\bmoney market\b|\bcash management\b", "money market"),
        ]
        for pattern, replacement in substitution_rules:
            normalized = re.sub(pattern, replacement, normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def search(self, news_headline: str, top_k: int = 5) -> List[Dict[str, float]]:
        normalized_query = (
            self._canonicalize_query(news_headline) if self.apply_query_canonicalization else news_headline
        )
        query_text = f"query: {normalized_query}" if self.preset.bi_encoder.startswith("intfloat/e5") else normalized_query
        query_emb = self.bi_encoder.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        retrieve_n = min(len(self.docs_df), max(top_k * self.preset.retrieval_multiplier, top_k))
        hits = util.semantic_search(query_emb, self.corpus_embeddings, top_k=retrieve_n)[0]

        scored_results: List[Dict[str, float]] = []
        cross_scores: Dict[int, float] = {}
        if self.cross_encoder is not None and self.cross_encoder_top_n > 0:
            top_for_cross = hits[: min(self.cross_encoder_top_n, len(hits))]
            pairs = [[normalized_query, self.docs_df.iloc[hit["corpus_id"]]["text"]] for hit in top_for_cross]
            if pairs:
                predictions = self.cross_encoder.predict(pairs)
                for hit, raw_score in zip(top_for_cross, predictions):
                    cross_scores[hit["corpus_id"]] = self._sigmoid(float(raw_score))

        for hit in hits:
            corpus_id = int(hit["corpus_id"])
            row = self.docs_df.iloc[corpus_id]
            bi_score = float(hit["score"])
            cross_score = cross_scores.get(corpus_id, 0.0)
            meta_boost = self._calculate_boost(normalized_query, row)
            final_score = bi_score + meta_boost + (self.preset.cross_encoder_weight * cross_score)

            scored_results.append(
                {
                    "ticker": str(row["ticker"]),
                    "text": str(row["text"]),
                    "final_score": float(final_score),
                    "bi_score": float(bi_score),
                    "cross_score": float(cross_score),
                    "boost": float(meta_boost),
                    "corpus_mode": self.corpus_mode,
                }
            )

        scored_results.sort(key=lambda item: (-item["final_score"], item["ticker"]))
        return scored_results[:top_k]

    def evaluate(self, labelled_examples: List[Dict[str, object]], top_k: int = 3) -> Dict[str, object]:
        if not labelled_examples:
            raise ValueError("labelled_examples cannot be empty")

        top1_hits = 0
        topk_hits = 0
        labelled_count = 0
        elapsed: List[float] = []
        details: List[Dict[str, object]] = []

        for example in labelled_examples:
            query = str(example["query"])
            raw_relevant = example.get("relevant_tickers", [])
            gold = {str(code).zfill(5) for code in raw_relevant if str(code).strip()}
            is_labelled = len(gold) > 0
            start = time.perf_counter()
            results = self.search(query, top_k=top_k)
            latency = (time.perf_counter() - start) * 1000.0
            elapsed.append(latency)

            predicted = [result["ticker"] for result in results]
            if is_labelled:
                labelled_count += 1
                if predicted and predicted[0] in gold:
                    top1_hits += 1
                if any(ticker in gold for ticker in predicted):
                    topk_hits += 1

            details.append(
                {
                    "query": query,
                    "relevant_tickers": sorted(gold),
                    "predicted_top_k": predicted,
                    "latency_ms": round(latency, 2),
                }
            )

        count = len(labelled_examples)
        top1_precision = round(top1_hits / labelled_count, 4) if labelled_count else None
        topk_recall = round(topk_hits / labelled_count, 4) if labelled_count else None
        return {
            "preset": self.preset.name,
            "corpus_mode": self.corpus_mode,
            "examples": count,
            "labelled_examples": labelled_count,
            "top1_precision": top1_precision,
            "topk_recall": topk_recall,
            "avg_latency_ms": round(sum(elapsed) / count, 2),
            "p95_latency_ms": round(sorted(elapsed)[max(int(count * 0.95) - 1, 0)], 2),
            "details": details,
        }


DEFAULT_NEWS_EXAMPLES: List[Dict[str, object]] = [
    {
        "query": "HKMA interest rate decision impacts Hong Kong money market and short-term liquidity products",
    },
    {
        "query": "Oil price shock hits commodity funds with increased volatility and roll risk",
    },
    {
        "query": "Hang Seng tech selloff pressures Hong Kong equity market with growth-heavy exposure",
    },
    {
        "query": "US labor data surprise shifts bond yields and affects fixed income performance",
    },
    {
        "query": "China policy easing could support mainland focused equities in the near term",
    },
    {
        "query": "Fed rate cut expectations influence short-duration bonds and cash-management products",
    },
    {
        "query": "RMB depreciation risk rises for mainland China equities with currency sensitivity",
    },
    {
        "query": "Dividend strategy under pressure in high-yield Asia equity income products",
    },
]


def _default_paths() -> Dict[str, str]:
    data_root = Path(__file__).resolve().parents[3] / "data"
    model_output_root = Path(__file__).resolve().parents[3] / "model_output" / "synpse"
    etf_root = data_root / "etf" if (data_root / "etf").exists() else data_root / "ETF"
    summary_dir = etf_root / "summary" if (etf_root / "summary").exists() else etf_root / "Summary"
    return {
        "documentation_dir": str(etf_root / "documentation"),
        "metadata_excel": str(summary_dir / "ETP_Data_Export.xlsx"),
        "profile_csv": str(model_output_root / "etf_profiles.csv"),
        "cache_dir": str(model_output_root / "cache"),
    }


def run_side_by_side_evaluation(
    labelled_examples: Optional[List[Dict[str, object]]] = None,
    preset: str = "fast",
    top_k: int = 3,
    sentence_row_cap: int = 4000,
) -> Dict[str, object]:
    paths = _default_paths()
    dataset = labelled_examples or DEFAULT_NEWS_EXAMPLES
    profile_engine = ETFNewsEngine(
        **paths,
        preset=preset,
        corpus_mode="profile",
    )
    sentence_engine = ETFNewsEngine(
        **paths,
        preset=preset,
        corpus_mode="sentence",
        sentence_row_cap=sentence_row_cap,
    )
    return {
        "sentence_row_cap": sentence_row_cap,
        "profile": profile_engine.evaluate(dataset, top_k=top_k),
        "sentence": sentence_engine.evaluate(dataset, top_k=top_k),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synapse ETF semantic matching engine")
    parser.add_argument("--query", type=str, default="", help="News headline/query for search")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results")
    parser.add_argument("--preset", type=str, default="fast", choices=sorted(MODEL_PRESETS.keys()))
    parser.add_argument("--corpus-mode", type=str, default="profile", choices=["profile", "sentence"])
    parser.add_argument("--enable-cross-encoder", action="store_true", help="Enable cross encoder reranking")
    parser.add_argument("--disable-cross-encoder", action="store_true", help="Disable cross encoder reranking")
    parser.add_argument("--cross-top-n", type=int, default=None, help="Number of candidates for cross encoder")
    parser.add_argument("--run-benchmark", action="store_true", help="Run side-by-side profile vs sentence evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.enable_cross_encoder and args.disable_cross_encoder:
        raise ValueError("Cannot set both --enable-cross-encoder and --disable-cross-encoder")

    if args.run_benchmark:
        benchmark = run_side_by_side_evaluation(preset=args.preset, top_k=max(args.top_k, 3))
        print(json.dumps(benchmark, indent=2))
    else:
        options = _default_paths()
        if args.enable_cross_encoder:
            cross_flag = True
        elif args.disable_cross_encoder:
            cross_flag = False
        else:
            cross_flag = None

        engine = ETFNewsEngine(
            **options,
            preset=args.preset,
            corpus_mode=args.corpus_mode,
            use_cross_encoder=cross_flag,
            cross_encoder_top_n=args.cross_top_n,
        )
        query = args.query or "HKMA interest rate changes affect Hong Kong money market ETFs"
        for result in engine.search(query, top_k=args.top_k):
            print(
                f"[{result['ticker']}] score={result['final_score']:.4f} "
                f"(bi={result['bi_score']:.4f}, cross={result['cross_score']:.4f}, boost={result['boost']:.4f})"
            )
            print(result["text"][:240] + "...")