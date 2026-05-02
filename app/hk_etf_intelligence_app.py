"""
HK ETF Intelligence Platform (Streamlit).

Run:
    streamlit run app/hk_etf_intelligence_app.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.synthesis.synthesis_engine import (  # noqa: E402
    DEFAULT_LOCAL_QWEN_MODEL,
    generate_synthesis,
    warmup_synthesis_model,
)

RAW_METADATA_COLUMNS = [
    "Stock code*",
    "Stock short name*",
    "Listing date*",
    "Product sub-category*",
    "Dividend yield (%)*",
    "Ongoing Charges Figures (%)*",
    "AUM",
    "Closing price",
    "Premium/discount %",
    "Asset class*",
    "Geographic focus*",
    "Investment focus*",
    "Management Style",
    "Thematic",
]

DISPLAY_COLUMN_MAP = {
    "Stock code*": "Ticker",
    "Stock short name*": "ETF Name",
    "Asset class*": "Asset Class",
    "Geographic focus*": "Geographic Focus",
    "Investment focus*": "Investment Focus",
    "AUM": "AUM",
    "Closing price": "Closing Price",
    "Premium/discount %": "Premium/Discount (%)",
    "Dividend yield (%)*": "Dividend Yield (%)",
    "Ongoing Charges Figures (%)*": "Ongoing Charges (%)",
    "Listing date*": "Listing Date",
    "Product sub-category*": "Sub-Category",
    "Management Style": "Management Style",
    "Thematic": "Thematic",
}


def apply_frontend_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Azeret+Mono:wght@400;600;700&family=Manrope:wght@400;500;700;800&display=swap');

        :root {
            --bg0: #082032;
            --bg1: #0f3a52;
            --panel: rgba(9, 37, 54, 0.74);
            --panel-border: rgba(98, 242, 201, 0.22);
            --ink: #eef8ff;
            --muted: #a8c8da;
            --cyan: #62f2c9;
            --amber: #ffbe4f;
            --rose: #ff6a92;
        }

        .stApp {
            font-family: "Manrope", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(1200px 420px at 15% -10%, rgba(98, 242, 201, 0.22), transparent 60%),
                radial-gradient(900px 360px at 95% 0%, rgba(255, 190, 79, 0.16), transparent 60%),
                radial-gradient(1000px 420px at 50% 115%, rgba(98, 170, 255, 0.16), transparent 60%),
                linear-gradient(135deg, var(--bg0), var(--bg1));
        }

        h1, h2, h3, .stMarkdown strong {
            font-family: "Azeret Mono", monospace;
            letter-spacing: 0.01em;
        }

        .hero-wrap {
            margin: 0.2rem 0 1rem 0;
            padding: 1.1rem 1.2rem;
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            background:
                linear-gradient(140deg, rgba(98, 242, 201, 0.13), rgba(98, 242, 201, 0.02) 45%, rgba(255, 190, 79, 0.09)),
                var(--panel);
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.35);
            animation: riseIn 900ms ease-out;
        }

        .hero-title {
            margin: 0;
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--ink);
            text-transform: uppercase;
        }

        .hero-sub {
            margin-top: 0.45rem;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .hero-kicker {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 207, 110, 0.45);
            color: var(--amber);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
        }

        [data-testid="stTabs"] {
            background: rgba(8, 23, 38, 0.55);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 0.35rem 0.65rem 0.1rem 0.65rem;
            animation: riseIn 1100ms ease-out;
        }

        [data-testid="stTab"] {
            font-family: "Azeret Mono", monospace;
            color: var(--muted);
        }

        [data-testid="stTab"][aria-selected="true"] {
            color: var(--cyan);
        }

        .stButton button {
            border-radius: 999px;
            border: 1px solid rgba(123, 247, 216, 0.42);
            color: #06131f;
            font-weight: 700;
            background: linear-gradient(120deg, var(--cyan), #8de7ff);
            transition: transform 150ms ease, box-shadow 150ms ease;
        }

        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(123, 247, 216, 0.32);
        }

        [data-testid="stMetric"] {
            border-radius: 14px;
            border: 1px solid var(--panel-border);
            padding: 0.35rem 0.6rem;
            background: rgba(10, 43, 62, 0.62);
        }

        [data-testid="stDataFrame"], .stPlotlyChart, [data-testid="stChatMessage"] {
            border-radius: 14px;
            border: 1px solid rgba(98, 242, 201, 0.18);
            background: rgba(8, 33, 49, 0.62);
        }

        /* Gemini-like chat UX */
        [data-testid="stChatMessage"] {
            border-radius: 20px;
            padding: 0.4rem 0.65rem;
            margin-bottom: 0.55rem;
            border: 1px solid rgba(140, 201, 228, 0.25);
            backdrop-filter: blur(4px);
        }
        [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
            border-radius: 999px;
        }
        [data-testid="stChatInput"] {
            border: 1px solid rgba(140, 201, 228, 0.30);
            border-radius: 18px;
            background: rgba(10, 37, 54, 0.88);
        }
        [data-testid="stChatInput"] textarea {
            font-size: 0.98rem;
        }

        @keyframes riseIn {
            from { transform: translateY(8px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(tickers: list[str], metadata: pd.DataFrame) -> None:
    total_etf = len(tickers)
    loaded_meta = len(metadata) if not metadata.empty else 0
    st.markdown(
        f"""
        <div class="hero-wrap">
            <p class="hero-title">HK ETF Financial Intelligence Terminal ✨</p>
            <p class="hero-sub">
                Screen opportunities, track weekly momentum, inspect holdings, and talk to your synthesis model.
                Built for retail investors who want institutional-style context without the noise.
            </p>
            <span class="hero-kicker">LIVE CONTEXT · {total_etf} TRACKED · {loaded_meta} METADATA ROWS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _default_etf_root() -> Path:
    data_root = PROJECT_ROOT / "data"
    return data_root / "etf" if (data_root / "etf").exists() else data_root / "ETF"


def _to_hkex_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    return digits.zfill(4)


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.2f}%"


def _format_pct_1dp(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1f}%"


def _bucket_asset_class(asset_class: object) -> str:
    text = str(asset_class).strip().lower()
    if not text or text == "nan":
        return "Other"
    if "equity" in text:
        return "Equity"
    if "commodity" in text:
        return "Commodity"
    if "fixed income" in text or "bond" in text or "money market" in text:
        return "Fixed Income"
    return "Other"


def _looks_like_etf_subcategory(value: object) -> bool:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return False
    return ("etf" in text) or ("exchange traded fund" in text)


def _query_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", str(text).lower()))


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = str(text).lower()
    return any(k in lowered for k in keywords)


def _is_recommendation_query(text: str) -> bool:
    return _contains_any(
        text,
        [
            "which etf",
            "recommend",
            "suggest",
            "invest in",
            "what etf",
            "可什麼etf",
            "咩etf",
            "推薦",
            "建議",
            "投資",
            "邊隻etf",
        ],
    )


def _extract_query_profile(query: str) -> dict[str, Any]:
    q = str(query).lower()
    theme_map: dict[str, list[str]] = {
        "oil": ["oil", "crude", "wti", "brent", "石油", "原油", "能源"],
        "gold": ["gold", "bullion", "黃金"],
        "china_tech": ["china tech", "chinese tech", "中國科技", "科技股", "科網", "internet", "platform"],
        "dividend": ["dividend", "yield", "income", "派息", "股息", "收益"],
        "bond": ["bond", "fixed income", "treasury", "債券", "債", "固定收益"],
        "money_market": ["money market", "cash management", "貨幣市場", "現金管理"],
    }
    matched_themes = [theme for theme, kws in theme_map.items() if any(k in q for k in kws)]
    return {"themes": matched_themes, "query_tokens": _query_tokens(query)}


@st.cache_data(show_spinner=False)
def load_risk_profile_corpus() -> pd.DataFrame:
    profile_path = PROJECT_ROOT / "model_output" / "synpse" / "etf_profiles.csv"
    if not profile_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(profile_path).copy()
    except Exception:
        return pd.DataFrame()
    if "ticker" not in df.columns:
        return pd.DataFrame()
    df["ticker"] = df["ticker"].map(_to_hkex_code)
    text_col = "key_risks" if "key_risks" in df.columns else "profile_text"
    if text_col not in df.columns:
        return pd.DataFrame()
    df["risk_text"] = df[text_col].fillna("").astype(str).str.strip()
    df = df[(df["ticker"] != "") & (df["ticker"] != "0000") & (df["risk_text"] != "")]
    return df[["ticker", "risk_text"]].drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_risk_profile_details() -> pd.DataFrame:
    profile_path = PROJECT_ROOT / "model_output" / "synpse" / "etf_profiles.csv"
    if not profile_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(profile_path).copy()
    except Exception:
        return pd.DataFrame()
    if "ticker" not in df.columns:
        return pd.DataFrame()

    df["ticker"] = df["ticker"].map(_to_hkex_code)
    for col in ["benchmark_or_index", "key_risks"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[(df["ticker"] != "") & (df["ticker"] != "0000")]
    return (
        df[["ticker", "benchmark_or_index", "key_risks"]]
        .drop_duplicates(subset=["ticker"], keep="first")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_mappable_ticker_universe() -> set[str]:
    """
    Tickers that are fully mappable by local recommendation/answer pipeline.
    Current rule: must exist in both per-ticker QA corpus and risk-profile corpus.
    """
    qa_root = PROJECT_ROOT / "model_output" / "Synthesis" / "finetune_all" / "per_ticker"
    qa_tickers: set[str] = set()
    if qa_root.exists():
        for path in qa_root.glob("*_finetune_qa.csv"):
            stem = path.name.replace("_finetune_qa.csv", "")
            code = _to_hkex_code(stem)
            if code and code != "0000":
                qa_tickers.add(code)

    profile = load_risk_profile_corpus()
    profile_tickers = set(profile["ticker"].astype(str).tolist()) if not profile.empty else set()

    if qa_tickers and profile_tickers:
        return qa_tickers & profile_tickers
    if qa_tickers:
        return qa_tickers
    return profile_tickers


@st.cache_resource(show_spinner=False)
def _load_sentence_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def _build_risk_profile_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> tuple[list[str], np.ndarray]:
    corpus = load_risk_profile_corpus()
    if corpus.empty:
        return [], np.empty((0, 0), dtype=np.float32)
    encoder = _load_sentence_encoder(model_name)
    vectors = encoder.encode(
        corpus["risk_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(vectors, dtype=np.float32)
    return corpus["ticker"].tolist(), embeddings


@st.cache_data(show_spinner=False)
def load_latest_synapse_topk() -> pd.DataFrame:
    syn_root = PROJECT_ROOT / "model_output" / "synpse"
    files = sorted(syn_root.glob("news_events_run_*/news_event_topk_matches.csv"))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[-1]).copy()
    except Exception:
        return pd.DataFrame()
    if "predicted_ticker" not in df.columns:
        return pd.DataFrame()
    df["predicted_ticker"] = df["predicted_ticker"].map(_to_hkex_code)
    return df


def rank_etf_candidates_hybrid(
    query: str,
    metadata: pd.DataFrame,
    tickers: list[str],
    max_items: int = 5,
) -> list[dict[str, Any]]:
    if not query or not tickers:
        return []

    mappable = load_mappable_ticker_universe()
    allowed = set(tickers)
    if mappable:
        allowed = allowed & mappable
    if not allowed:
        return []
    profile = _extract_query_profile(query)
    q_tokens = set(profile.get("query_tokens", set()))
    themes = profile.get("themes", [])

    # 1) Metadata/theme score (primary)
    base_rows: list[dict[str, Any]] = []
    if not metadata.empty and "Ticker" in metadata.columns:
        cols = [
            c
            for c in [
                "Stock short name*",
                "Asset class*",
                "Geographic focus*",
                "Investment focus*",
                "Product sub-category*",
                "Thematic",
            ]
            if c in metadata.columns
        ]
        work = metadata[metadata["Ticker"].isin(allowed)].copy()
        for _, row in work.iterrows():
            code = _to_hkex_code(row.get("Ticker", ""))
            if not code or code == "0000":
                continue
            bundle = " ".join(str(row.get(c, "")) for c in cols).lower()
            r_tokens = _query_tokens(bundle)
            token_overlap = len(q_tokens & r_tokens) / max(len(q_tokens), 1) if q_tokens else 0.0
            theme_hits = 0.0
            if themes:
                if "oil" in themes and _contains_any(bundle, ["oil", "crude", "wti", "brent", "石油", "原油", "能源"]):
                    theme_hits += 1.0
                if "gold" in themes and _contains_any(bundle, ["gold", "bullion", "黃金"]):
                    theme_hits += 1.0
                if "china_tech" in themes and _contains_any(bundle, ["china", "chinese", "tech", "科技", "科網", "internet"]):
                    theme_hits += 1.0
                if "dividend" in themes and _contains_any(bundle, ["dividend", "yield", "income", "派息", "股息"]):
                    theme_hits += 1.0
                if "bond" in themes and _contains_any(bundle, ["bond", "fixed income", "treasury", "債"]):
                    theme_hits += 1.0
                if "money_market" in themes and _contains_any(bundle, ["money market", "cash", "貨幣市場"]):
                    theme_hits += 1.0
            theme_score = theme_hits / max(len(themes), 1) if themes else 0.0
            base_rows.append(
                {
                    "ticker": code,
                    "name": str(row.get("Stock short name*", "")).strip() or "Unknown ETF",
                    "metadata_score": float(token_overlap),
                    "theme_score": float(theme_score),
                }
            )

    # 2) Risk-profile cosine signal
    risk_candidates = infer_ticker_candidates_from_risk_profile(query, tickers, max_items=30)
    risk_map = {code: float(score) for code, score in risk_candidates}
    max_risk = max(risk_map.values()) if risk_map else 1.0

    # 3) Optional news auxiliary (small weight only)
    news_map: dict[str, float] = {}
    syn = load_latest_synapse_topk()
    if not syn.empty:
        work = syn[syn["predicted_ticker"].isin(allowed)].copy()
        text_cols = [c for c in ["Headline", "Market_Event", "query_text", "Sector", "Source"] if c in work.columns]
        if not work.empty and text_cols and q_tokens:
            def _aux_score(row: pd.Series) -> float:
                content = " ".join(str(row.get(c, "")) for c in text_cols)
                overlap = len(q_tokens & _query_tokens(content)) / max(len(q_tokens), 1)
                sim = float(pd.to_numeric(row.get("final_score", 0.0), errors="coerce") or 0.0)
                return 0.6 * overlap + 0.4 * sim

            work["aux_score"] = work.apply(_aux_score, axis=1)
            grouped = work.groupby("predicted_ticker", as_index=False)["aux_score"].max()
            news_map = {str(r["predicted_ticker"]): float(r["aux_score"]) for _, r in grouped.iterrows()}

    # Merge universe from metadata + risk candidates.
    scored: dict[str, dict[str, Any]] = {r["ticker"]: r for r in base_rows}
    for code in risk_map:
        if code not in scored:
            scored[code] = {"ticker": code, "name": "Unknown ETF", "metadata_score": 0.0, "theme_score": 0.0}

    # Fill missing names from metadata lookup.
    if not metadata.empty and "Ticker" in metadata.columns and "Stock short name*" in metadata.columns:
        name_map = metadata[["Ticker", "Stock short name*"]].drop_duplicates().set_index("Ticker")["Stock short name*"].to_dict()
        for code, item in scored.items():
            if item.get("name") == "Unknown ETF":
                item["name"] = str(name_map.get(code, "Unknown ETF"))

    ranked: list[dict[str, Any]] = []
    for code, item in scored.items():
        risk_score = risk_map.get(code, 0.0)
        risk_norm = risk_score / max(max_risk, 1e-6)
        aux_score = news_map.get(code, 0.0)
        final_score = (
            0.40 * float(item.get("theme_score", 0.0))
            + 0.30 * float(item.get("metadata_score", 0.0))
            + 0.25 * float(risk_norm)
            + 0.05 * float(aux_score)
        )
        ranked.append(
            {
                "ticker": code,
                "name": item.get("name", "Unknown ETF"),
                "final_score": float(final_score),
                "theme_score": float(item.get("theme_score", 0.0)),
                "metadata_score": float(item.get("metadata_score", 0.0)),
                "risk_score": float(risk_score),
                "news_aux_score": float(aux_score),
            }
        )
    ranked.sort(key=lambda x: (-x["final_score"], x["ticker"]))
    return ranked[:max_items]


def infer_ticker_from_query(query: str, metadata: pd.DataFrame, tickers: list[str]) -> tuple[str | None, str]:
    """
    Infer a likely ETF ticker from free-form user query.
    Priority:
    1) explicit 4-digit ticker in query
    2) ETF short name substring match in metadata
    3) no confident match -> None
    """
    if not query:
        return None, "No query provided."
    mappable = load_mappable_ticker_universe()
    eligible = set(tickers)
    if mappable:
        eligible = eligible & mappable
    if not eligible:
        return None, "No mappable ETF universe available."

    # 1) explicit ticker in query
    ticker_hits = re.findall(r"\b\d{4}\b", query)
    for hit in ticker_hits:
        if hit in eligible and hit != "0000":
            return hit, f"Detected explicit ticker `{hit}` from your question."

    # 2) metadata name match
    if not metadata.empty and "Ticker" in metadata.columns and "Stock short name*" in metadata.columns:
        q = query.lower()
        candidates = metadata[["Ticker", "Stock short name*"]].dropna().copy()
        candidates["name_lower"] = candidates["Stock short name*"].astype(str).str.lower()

        contains = candidates[candidates["name_lower"].apply(lambda name: name in q or q in name)]
        contains = contains[contains["Ticker"].isin(eligible)]
        contains = contains[contains["Ticker"] != "0000"]
        if not contains.empty:
            best = str(contains.iloc[0]["Ticker"])
            name = str(contains.iloc[0]["Stock short name*"])
            return best, f"Matched ETF name `{name}` -> ticker `{best}`."

    # 3) hybrid inference for free-form/no-ticker queries
    ranked = rank_etf_candidates_hybrid(query, metadata, tickers, max_items=3)
    if ranked:
        top_score = float(ranked[0]["final_score"])
        second_score = float(ranked[1]["final_score"]) if len(ranked) > 1 else 0.0
        if top_score >= 0.20 and (top_score - second_score) >= 0.03:
            top_code = str(ranked[0]["ticker"])
            return top_code, f"Inferred from hybrid profile matching -> `{top_code}`."

    return None, "No confident ticker match found from query text."


def infer_ticker_candidates_from_risk_profile(
    query: str,
    tickers: list[str],
    max_items: int = 3,
) -> list[tuple[str, float]]:
    """
    Return ranked ticker candidates from risk-profile cosine similarity.
    """
    if not query or not tickers:
        return []
    ticker_list, embeddings = _build_risk_profile_embeddings()
    if not ticker_list or embeddings.size == 0:
        return []

    allowed = set(tickers)
    keep_idx = [i for i, code in enumerate(ticker_list) if code in allowed]
    if not keep_idx:
        return []

    filtered_tickers = [ticker_list[i] for i in keep_idx]
    filtered_embeddings = embeddings[keep_idx]

    try:
        encoder = _load_sentence_encoder()
        query_vec = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        query_vec = np.asarray(query_vec, dtype=np.float32)
        scores = filtered_embeddings @ query_vec
    except Exception:
        # Fallback lexical matching if embedding model unavailable at runtime.
        corpus = load_risk_profile_corpus()
        corpus = corpus[corpus["ticker"].isin(allowed)].copy()
        if corpus.empty:
            return []
        q_tokens = _query_tokens(query)
        if not q_tokens:
            return []
        corpus["score"] = corpus["risk_text"].map(
            lambda txt: len(q_tokens & _query_tokens(txt)) / max(len(q_tokens), 1)
        )
        ranked = corpus.sort_values("score", ascending=False).head(max_items)
        return [(str(row["ticker"]), float(row["score"])) for _, row in ranked.iterrows() if float(row["score"]) > 0]

    order = np.argsort(-scores)
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for idx in order:
        code = filtered_tickers[int(idx)]
        if code in seen:
            continue
        seen.add(code)
        score = float(scores[int(idx)])
        if score <= 0:
            continue
        out.append((code, score))
        if len(out) >= max_items:
            break
    return out


def build_etf_high_level_summary(ticker: str, metadata: pd.DataFrame) -> str:
    """
    Build a concise ETF profile summary for chatbot context display.
    """
    code = _to_hkex_code(ticker)
    if not code:
        return "ETF profile summary unavailable."

    if metadata.empty or "Ticker" not in metadata.columns:
        return f"ETF `{code}` selected. Name unavailable from current metadata."

    match = metadata.loc[metadata["Ticker"] == code]
    if match.empty:
        return f"ETF `{code}` selected. Name unavailable from current metadata."

    row = match.iloc[0]
    etf_name = str(row.get("Stock short name*", "")).strip() or "Unknown ETF"
    asset_class = str(row.get("Asset class*", "")).strip() or "-"
    geo_focus = str(row.get("Geographic focus*", "")).strip() or "-"
    invest_focus = str(row.get("Investment focus*", "")).strip() or "-"
    mgmt_style = str(row.get("Management Style", "")).strip() or "-"

    return (
        f"**ETF Name:** {etf_name}  \n"
        f"**Ticker:** `{code}`  \n"
        f"**Profile:** {asset_class} | {geo_focus} | {invest_focus} | {mgmt_style}"
    )


def build_recommendation_message(query: str, ranked: list[dict[str, Any]], metadata: pd.DataFrame) -> str:
    if not ranked:
        return (
            "I couldn't confidently recommend a single ETF from your query. "
            "Please add a bit more preference (e.g., region, volatility tolerance, or asset class)."
        )

    primary = ranked[0]
    primary_ticker = str(primary["ticker"])
    primary_name = str(primary.get("name", "Unknown ETF"))
    summary = build_etf_high_level_summary(primary_ticker, metadata)

    benchmark = "-"
    key_risks = "-"
    details = load_risk_profile_details()
    if not details.empty:
        hit = details.loc[details["ticker"] == primary_ticker]
        if not hit.empty:
            bench_txt = str(hit.iloc[0].get("benchmark_or_index", "")).strip()
            risk_txt = str(hit.iloc[0].get("key_risks", "")).strip()
            if bench_txt and bench_txt.lower() != "nan":
                benchmark = bench_txt
            if risk_txt and risk_txt.lower() != "nan":
                # Keep recommendation response concise and scannable.
                key_risks = re.sub(r"\s+", " ", risk_txt)
                if len(key_risks) > 200:
                    key_risks = key_risks[:200].rsplit(" ", 1)[0] + "..."

    fee_text = "-"
    if not metadata.empty and "Ticker" in metadata.columns:
        fee_row = metadata.loc[metadata["Ticker"] == primary_ticker]
        if not fee_row.empty and "Ongoing Charges Figures (%)*" in fee_row.columns:
            fee_val = pd.to_numeric(fee_row.iloc[0]["Ongoing Charges Figures (%)*"], errors="coerce")
            if pd.notna(fee_val):
                fee_text = f"{float(fee_val):.2f}%"

    alt = [f"`{x['ticker']}` {x.get('name', 'Unknown ETF')}" for x in ranked[1:4]]
    alt_line = ", ".join(alt) if alt else "No close alternatives found."

    return (
        f"`{primary_ticker}` ({primary_name}) is the best match for your request.\n\n"
        "It provides exposure aligned with the theme in your query. "
        "Here is a high-level ETF summary:\n\n"
        f"{summary}\n\n"
        f"**Benchmark:** {benchmark}\n\n"
        f"**Key risks:** {key_risks}\n\n"
        f"**Ongoing fee (OCF):** {fee_text}\n\n"
        f"You could also explore: {alt_line}."
    )


def suggest_etf_candidates_from_query(
    query: str,
    metadata: pd.DataFrame,
    max_items: int = 5,
) -> list[tuple[str, str]]:
    """
    Return a small ranked ETF candidate list from metadata text overlap.
    """
    if not query or metadata.empty or "Ticker" not in metadata.columns:
        return []

    work = metadata.copy()
    text_cols = [
        col
        for col in [
            "Stock short name*",
            "Asset class*",
            "Geographic focus*",
            "Investment focus*",
            "Product sub-category*",
            "Thematic",
        ]
        if col in work.columns
    ]
    if not text_cols:
        return []

    q_tokens = _query_tokens(query)
    if not q_tokens:
        return []

    def _cand_score(row: pd.Series) -> float:
        content = " ".join(str(row.get(c, "")) for c in text_cols)
        r_tokens = _query_tokens(content)
        if not r_tokens:
            return 0.0
        return len(q_tokens & r_tokens) / max(len(q_tokens), 1)

    work["query_match_score"] = work.apply(_cand_score, axis=1)
    ranked = (
        work[work["query_match_score"] > 0]
        .sort_values(["query_match_score", "Ticker"], ascending=[False, True])
        .drop_duplicates(subset=["Ticker"], keep="first")
        .head(max_items)
    )
    if ranked.empty:
        return []

    out: list[tuple[str, str]] = []
    for _, row in ranked.iterrows():
        code = _to_hkex_code(row.get("Ticker", ""))
        if not code or code == "0000":
            continue
        name = str(row.get("Stock short name*", "")).strip() or "Unknown ETF"
        out.append((code, name))
    return out


def _is_timeout_fallback_response(text: str) -> bool:
    body = (text or "").lower()
    return (
        "could not produce a full response in time" in body
        or "concise factual fallback" in body
    )


def _fix_invalid_ticker_claim(response: str, resolved_ticker: str | None, tickers: list[str], metadata: pd.DataFrame) -> str:
    """
    Guardrail against model hallucination: claiming a valid resolved ticker is invalid.
    """
    text = str(response or "").strip()
    code = _to_hkex_code(resolved_ticker or "")
    if not text or not code:
        return text

    lower = text.lower()
    is_known_ticker = code in set(tickers)
    if not is_known_ticker:
        return text

    invalid_patterns = [
        "does not correspond to any known etf",
        "does not correspond to a known etf",
        "does not correspond to an existing etf",
        "is not a known etf",
        "not an existing etf",
        "isn't an etf",
        "is not in our database",
        "isn't in our database",
        "there isn't an etf with the ticker",
        "there is no etf with the ticker",
        "not found in our database",
        "couldn't find any information about the etf with the ticker",
        "could not find any information about the etf with the ticker",
        "might be incorrect",
        "not listed in the hong kong market",
        "confirm if",
        "correct ticker",
        "check the ticker again",
        "provide more details",
        "would you like more",
        "並不是一個現有的etf",
        "不是一個現有的etf",
        "未能找到對應的etf",
    ]
    # Also sanitize obvious truncation tails even when invalid-ETF phrase is absent.
    if "would you like more" in lower:
        text = re.sub(r"(?is)\bwould you like more.*$", "", text).strip()
        lower = text.lower()

    if not any(p in lower for p in invalid_patterns):
        return text

    # Replace hallucinated invalid-ticker claim with a concise correction.
    corrected_prefix = f"Using ticker `{code}` as the active ETF context.\n\n"

    # Drop common "ticker not found/invalid/please confirm ticker" sentence variants.
    cleaned = text
    sentence_patterns = [
        r'[^.\n]*there isn[\'’]t an etf with the ticker[^.\n]*[.\n]?',
        r'[^.\n]*there is no etf with the ticker[^.\n]*[.\n]?',
        r'[^.\n]*does not correspond to any known etf[^.\n]*[.\n]?',
        r'[^.\n]*does not correspond to an existing etf[^.\n]*[.\n]?',
        r'[^.\n]*isn[\'’]t in our database[^.\n]*[.\n]?',
        r'[^.\n]*is not in our database[^.\n]*[.\n]?',
        r'[^.\n]*couldn[\'’]t find any information about the etf with the ticker[^.\n]*[.\n]?',
        r'[^.\n]*could not find any information about the etf with the ticker[^.\n]*[.\n]?',
        r'[^.\n]*might be incorrect[^.\n]*[.\n]?',
        r'[^.\n]*not listed in the hong kong market[^.\n]*[.\n]?',
        r'[^.\n]*confirm if[^.\n]*correct ticker[^.\n]*[.\n]?',
        r'[^.\n]*check the ticker again[^.\n]*[.\n]?',
        r'[^.\n]*provide more details[^.\n]*[.\n]?',
        r'(?is)\bwould you like more.*$',
        r'[^。\n]*不是一個現有的etf[^。\n]*[。\n]?',
        r'[^。\n]*並不是一個現有的etf[^。\n]*[。\n]?',
        r'[^。\n]*未能找到對應的etf[^。\n]*[。\n]?',
    ]
    for pattern in sentence_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Last-resort scrub: if any remaining sentence still says ticker is invalid,
    # remove those lines completely when ticker is already known/resolved.
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    safe_lines: list[str] = []
    for ln in lines:
        low = ln.lower()
        if (
            "ticker" in low
            and (
                "does not correspond" in low
                or "not listed" in low
                or "incorrect" in low
                or "not in our database" in low
                or "isn't an etf" in low
            )
        ):
            continue
        safe_lines.append(ln)
    cleaned = "\n".join(safe_lines).strip()

    if not cleaned:
        return corrected_prefix + "I can continue with risk, benchmark, fee, and alternatives for this ETF."
    return corrected_prefix + cleaned


def _sanitize_resolved_ticker_response(
    response: str,
    resolved_ticker: str | None,
    tickers: list[str],
    metadata: pd.DataFrame,
) -> str:
    """
    Sanitize hallucinated/low-quality fragments after ticker is already resolved.
    """
    text = str(response or "").strip()
    code = _to_hkex_code(resolved_ticker or "")
    if not text or not code or code not in set(tickers):
        return text

    # If local mapping artifacts exist, remove contradictory "insufficient local data" claims.
    has_profile = False
    details = load_risk_profile_details()
    if not details.empty and "ticker" in details.columns:
        has_profile = code in set(details["ticker"].astype(str).tolist())
    mappable = load_mappable_ticker_universe()
    has_mapping = (not mappable) or (code in mappable) or has_profile
    if has_mapping:
        insuff_patterns = [
            r"[^。\n]*目前本地資料不足以確認細節[^。\n]*[。\n]?",
            r"[^.\n]*local data is currently insufficient[^.\n]*[.\n]?",
            r"[^.\n]*insufficient local data[^.\n]*[.\n]?",
        ]
        for p in insuff_patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE).strip()

    # Trim common truncated tail fragments.
    text = re.sub(r"(?is)(\(|（)\s*key\s*$", "", text).strip()
    text = re.sub(r"(?is)\bwould you like more.*$", "", text).strip()
    return text


def _warmup_model_once(backend: str, model_name: str, use_response_cache: bool) -> None:
    warm_key = f"{backend}:{model_name}:{int(use_response_cache)}"
    warmed = st.session_state.get("model_warmup_keys", set())
    if warm_key in warmed:
        return

    with st.spinner("Initializing chatbot model..."):
        warmup_synthesis_model(
            backend=backend,
            qwen_model=model_name,
            enable_response_cache=use_response_cache,
        )
    warmed.add(warm_key)
    st.session_state.model_warmup_keys = warmed


@st.cache_data(show_spinner=False)
def load_metadata() -> tuple[pd.DataFrame, str]:
    etf_root = _default_etf_root()
    metadata_path = etf_root / "summary" / "ETP_Data_Export.xlsx"
    if not metadata_path.exists():
        return pd.DataFrame(), str(metadata_path)

    try:
        df = pd.read_excel(metadata_path, skipfooter=2)
    except Exception:
        df = pd.read_excel(metadata_path)

    available_cols = [c for c in RAW_METADATA_COLUMNS if c in df.columns]
    if not available_cols:
        return pd.DataFrame(), str(metadata_path)

    meta = df.loc[:, available_cols].copy()
    if "Stock code*" in meta.columns:
        meta["Ticker"] = meta["Stock code*"].map(_to_hkex_code)
    else:
        meta["Ticker"] = ""
    meta = meta[(meta["Ticker"] != "") & (meta["Ticker"] != "0000")].copy()

    for col in [
        "AUM",
        "Closing price",
        "Premium/discount %",
        "Dividend yield (%)*",
        "Ongoing Charges Figures(%)*",
        "Ongoing Charges Figures (%)*",
    ]:
        if col in meta.columns:
            meta[col] = _clean_numeric_series(meta[col])

    # Normalize inconsistent column name found in some exports.
    if "Ongoing Charges Figures (%)*" not in meta.columns and "Ongoing Charges Figures(%)*" in meta.columns:
        meta["Ongoing Charges Figures (%)*"] = meta["Ongoing Charges Figures(%)*"]

    if "Listing date*" in meta.columns:
        meta["Listing date*"] = pd.to_datetime(meta["Listing date*"], errors="coerce")

    if "Stock code*" in meta.columns:
        meta["Stock code*"] = meta["Ticker"]

    # Keep ETF rows only when product subtype information is available.
    # This prevents plain equities from leaking into selector options.
    if "Product sub-category*" in meta.columns:
        etf_mask = meta["Product sub-category*"].apply(_looks_like_etf_subcategory)
        if etf_mask.any():
            meta = meta[etf_mask].copy()

    return meta.drop_duplicates(subset=["Ticker"], keep="last"), str(metadata_path)


@st.cache_data(show_spinner=False)
def discover_tickers() -> list[str]:
    tickers: set[str] = set()
    metadata, _ = load_metadata()
    allowed_tickers: set[str] = set()
    if not metadata.empty and "Ticker" in metadata.columns:
        allowed_tickers = set(metadata["Ticker"].dropna().astype(str).tolist())
        tickers.update(allowed_tickers)

    ohlcv_root = _default_etf_root() / "ohlcv"
    if ohlcv_root.exists():
        for folder in ohlcv_root.iterdir():
            if folder.is_dir():
                code = _to_hkex_code(folder.name)
                if code and (not allowed_tickers or code in allowed_tickers):
                    tickers.add(code)

    return sorted(t for t in tickers if t != "0000")


@st.cache_data(show_spinner=False)
def load_ohlcv(ticker: str) -> pd.DataFrame:
    code = _to_hkex_code(ticker)
    if not code:
        return pd.DataFrame()

    ohlcv_root = _default_etf_root() / "ohlcv"
    candidates = [
        ohlcv_root / code / "ohlcv.parquet",
        ohlcv_root / f"{code}.parquet",
        ohlcv_root / code / "ohlcv.csv",
        ohlcv_root / f"{code}.csv",
    ]

    target = next((path for path in candidates if path.exists()), None)
    if target is None:
        return pd.DataFrame()

    if target.suffix.lower() == ".parquet":
        df = pd.read_parquet(target)
    else:
        df = pd.read_csv(target)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [lvl0 for lvl0, *_ in df.columns.tolist()]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    elif "Close" in df.columns:
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"])


@st.cache_data(show_spinner=False)
def build_weekly_snapshot(tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        ohlcv = load_ohlcv(ticker)
        if ohlcv.empty or len(ohlcv) < 6:
            continue
        latest_close = float(ohlcv["Close"].iloc[-1])
        prev_week_close = float(ohlcv["Close"].iloc[-6])
        one_week_return = ((latest_close / prev_week_close) - 1.0) * 100.0 if prev_week_close > 0 else float("nan")
        rows.append(
            {
                "Ticker": ticker,
                "Latest Date": ohlcv.index[-1].date().isoformat(),
                "Latest Close": latest_close,
                "1W Return (%)": one_week_return,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("1W Return (%)", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_one_year_prices(ticker: str) -> pd.DataFrame:
    ohlcv = load_ohlcv(ticker)
    if ohlcv.empty:
        return pd.DataFrame()
    series = ohlcv[["Close"]].copy().tail(252).reset_index()
    series.columns = ["Date", "Close"]
    return series


@st.cache_data(show_spinner=False)
def load_top_holdings(ticker: str) -> pd.DataFrame:
    code = _to_hkex_code(ticker)
    if not code:
        return pd.DataFrame()

    holdings_root = _default_etf_root() / "holdings" / "top10"
    candidates = [
        holdings_root / code / "top_holdings.parquet",
        holdings_root / str(int(code)) / "top_holdings.parquet",
        holdings_root / code / "top_holdings.csv",
        holdings_root / str(int(code)) / "top_holdings.csv",
    ]
    target = next((path for path in candidates if path.exists()), None)
    if target is None:
        return pd.DataFrame()

    if target.suffix.lower() == ".parquet":
        df = pd.read_parquet(target)
    else:
        df = pd.read_csv(target)
    if df.empty:
        return pd.DataFrame()

    # Resolve columns case-insensitively to avoid schema drift across sources.
    col_lookup = {str(c).strip().lower(): c for c in df.columns}

    symbol_col = next(
        (
            col_lookup[key]
            for key in ["symbol", "holding_symbol", "ticker", "security_symbol"]
            if key in col_lookup
        ),
        None,
    )
    name_col = next(
        (
            col_lookup[key]
            for key in ["name", "holding_name", "holdingname", "security_name"]
            if key in col_lookup
        ),
        None,
    )
    weight_col = next(
        (
            col_lookup[key]
            for key in [
                "holding percent",
                "holding_percent",
                "holdingpercent",
                "holding_pct",
                "weight",
                "weight_pct",
                "percent",
                "percentage",
            ]
            if key in col_lookup
        ),
        None,
    )

    if weight_col is None:
        for col in df.columns:
            col_low = str(col).strip().lower()
            if any(token in col_low for token in ["percent", "weight", "holding"]):
                weight_col = col
                break

    out = pd.DataFrame()
    if symbol_col:
        out["Symbol"] = df[symbol_col].astype(str)
    if name_col:
        out["Name"] = df[name_col].astype(str)
    if not {"Symbol", "Name"} & set(out.columns):
        # Last-resort fallback for unexpected schemas.
        out["Name"] = df.iloc[:, 0].astype(str)

    if weight_col:
        out["Weight (%)"] = _clean_numeric_series(df[weight_col]).astype("float64")
        if out["Weight (%)"].dropna().mean() <= 1.0:
            out["Weight (%)"] = out["Weight (%)"] * 100.0

    required_text_cols = [c for c in ["Symbol", "Name"] if c in out.columns]
    if required_text_cols:
        text_mask = out[required_text_cols].astype(str).apply(lambda s: s.str.strip()).ne("").any(axis=1)
        out = out[text_mask]
    return out.head(10)


def apply_metadata_filters(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        return meta

    st.markdown("#### ETF Screener Filters")
    c1, c2 = st.columns([2, 1])
    search = c1.text_input("Search ticker / ETF name", placeholder="e.g. 2800 or Tracker Fund")
    show_count = c2.number_input("Rows to show", min_value=10, max_value=1000, value=100, step=10)

    filtered = meta.copy()

    if search:
        q = search.strip().lower()
        if "Stock short name*" in filtered.columns:
            name_series = filtered["Stock short name*"].astype(str)
        else:
            name_series = pd.Series("", index=filtered.index)
        mask = filtered["Ticker"].astype(str).str.contains(q, case=False, regex=False) | name_series.str.contains(q, case=False, regex=False)
        filtered = filtered[mask]

    cat_cols = [
        "Asset class*",
        "Geographic focus*",
        "Investment focus*",
        "Management Style",
        "Thematic",
    ]
    for col in cat_cols:
        if col not in filtered.columns:
            continue
        options = sorted(x for x in filtered[col].dropna().astype(str).unique().tolist() if x and x.lower() != "nan")
        if options:
            selected = st.multiselect(f"{DISPLAY_COLUMN_MAP.get(col, col)}", options)
            if selected:
                filtered = filtered[filtered[col].astype(str).isin(selected)]

    numeric_filters = [
        "AUM",
        "Dividend yield (%)*",
        "Ongoing Charges Figures (%)*",
        "Premium/discount %",
    ]
    for col in numeric_filters:
        if col not in filtered.columns:
            continue
        values = pd.to_numeric(filtered[col], errors="coerce").dropna()
        if values.empty:
            continue
        low, high = float(values.min()), float(values.max())
        if low == high:
            continue
        selected_low, selected_high = st.slider(
            f"{DISPLAY_COLUMN_MAP.get(col, col)} range",
            min_value=low,
            max_value=high,
            value=(low, high),
        )
        filtered = filtered[(pd.to_numeric(filtered[col], errors="coerce") >= selected_low) & (pd.to_numeric(filtered[col], errors="coerce") <= selected_high)]

    filtered = filtered.copy().head(int(show_count))
    return filtered


def render_screener(metadata: pd.DataFrame) -> None:
    st.subheader("ETF Screener 🔎")
    if metadata.empty:
        _, metadata_path = load_metadata()
        st.warning(f"Metadata file not found or unreadable: `{metadata_path}`")
        return

    filtered = apply_metadata_filters(metadata)
    st.caption(f"Filtered ETFs: {len(filtered)}")

    display_cols = [c for c in RAW_METADATA_COLUMNS if c in filtered.columns]
    show = filtered.loc[:, display_cols].rename(columns=DISPLAY_COLUMN_MAP).copy()
    if "Listing Date" in show.columns:
        show["Listing Date"] = pd.to_datetime(show["Listing Date"], errors="coerce").dt.date
    st.dataframe(show, use_container_width=True, hide_index=True)

    ticker_options = filtered["Ticker"].dropna().astype(str).tolist()
    if not ticker_options:
        return
    selected = st.selectbox("Select ETF from screener", options=ticker_options, key="screener_pick")
    if st.button("Use this ETF in Explorer + Chatbot", type="primary"):
        st.session_state.selected_ticker = selected
        st.success(f"Selected `{selected}` for Explorer and Chatbot.")


def render_weekly_winners_losers(tickers: list[str], metadata: pd.DataFrame) -> None:
    st.subheader("Weekly Winners and Losers ⚡")
    snapshot = build_weekly_snapshot(tickers)
    if snapshot.empty:
        st.info("No OHLCV data available under `data/etf/ohlcv` (or `data/ETF/ohlcv`).")
        return

    if not metadata.empty and "Ticker" in metadata.columns and "Stock short name*" in metadata.columns:
        name_map = metadata[["Ticker", "Stock short name*"]].drop_duplicates().set_index("Ticker")["Stock short name*"]
        snapshot["ETF Name"] = snapshot["Ticker"].map(name_map).fillna("-")
    if not metadata.empty and "Ticker" in metadata.columns and "Asset class*" in metadata.columns:
        class_map = metadata[["Ticker", "Asset class*"]].drop_duplicates().set_index("Ticker")["Asset class*"]
        snapshot["Asset Class"] = snapshot["Ticker"].map(class_map).fillna("Other")
        snapshot["Category"] = snapshot["Asset Class"].map(_bucket_asset_class)
    else:
        snapshot["Category"] = "Other"

    cols = ["Ticker", "ETF Name", "Latest Date", "Latest Close", "1W Return (%)"]
    cols = [c for c in cols if c in snapshot.columns]
    gainers = snapshot.head(5).copy()
    losers = snapshot.tail(5).sort_values("1W Return (%)", ascending=True).copy()

    positive_ratio = (snapshot["1W Return (%)"] > 0).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("ETFs with 1W gain", f"{positive_ratio:.1f}%")
    c2.metric("Top weekly return", _format_pct_1dp(float(snapshot["1W Return (%)"].max())))
    c3.metric("Bottom weekly return", _format_pct_1dp(float(snapshot["1W Return (%)"].min())))

    left, right = st.columns(2)
    with left:
        st.markdown("#### 🏆 Top 5 Winners")
        winners_view = gainers[cols].copy()
        if "1W Return (%)" in winners_view.columns:
            winners_view["1W Return (%)"] = winners_view["1W Return (%)"].map(_format_pct_1dp)
        st.dataframe(winners_view, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### 📉 Top 5 Losers")
        losers_view = losers[cols].copy()
        if "1W Return (%)" in losers_view.columns:
            losers_view["1W Return (%)"] = losers_view["1W Return (%)"].map(_format_pct_1dp)
        st.dataframe(losers_view, use_container_width=True, hide_index=True)

    st.markdown("#### 🎯 Top 5 Winners/Losers by Category")
    for category in ["Equity", "Commodity", "Fixed Income"]:
        cat_df = snapshot[snapshot["Category"] == category].copy()
        if cat_df.empty:
            st.caption(f"{category}: no ETFs available in current data.")
            continue
        cat_winners = cat_df.nlargest(5, "1W Return (%)")
        cat_losers = cat_df.nsmallest(5, "1W Return (%)")
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{category} Winners 🚀**")
            winners_cat_view = cat_winners[cols].copy()
            if "1W Return (%)" in winners_cat_view.columns:
                winners_cat_view["1W Return (%)"] = winners_cat_view["1W Return (%)"].map(_format_pct_1dp)
            st.dataframe(winners_cat_view, use_container_width=True, hide_index=True)
        with right:
            st.markdown(f"**{category} Losers 🧊**")
            losers_cat_view = cat_losers[cols].copy()
            if "1W Return (%)" in losers_cat_view.columns:
                losers_cat_view["1W Return (%)"] = losers_cat_view["1W Return (%)"].map(_format_pct_1dp)
            st.dataframe(losers_cat_view, use_container_width=True, hide_index=True)

    st.markdown("#### Full Ranking")
    ranking = snapshot[cols].copy()
    if "1W Return (%)" in ranking.columns:
        ranking = ranking.style.format({"1W Return (%)": _format_pct_1dp}).map(
            lambda val: "color: #7bf7d8;" if isinstance(val, (int, float)) and val > 0 else "color: #ff6e8a;",
            subset=["1W Return (%)"],
        )
    st.dataframe(ranking, use_container_width=True, hide_index=True)


def render_explorer(tickers: list[str], metadata: pd.DataFrame) -> None:
    st.subheader("ETF Explorer 📈")
    if not tickers:
        st.info("No ETF tickers discovered from metadata or OHLCV directories.")
        return

    current = st.session_state.get("selected_ticker", tickers[0])
    if current not in tickers:
        current = tickers[0]

    selected_ticker = st.selectbox("Select ETF", options=tickers, index=tickers.index(current), key="explorer_ticker")
    st.session_state.selected_ticker = selected_ticker

    if not metadata.empty and "Ticker" in metadata.columns and "Stock short name*" in metadata.columns:
        match = metadata.loc[metadata["Ticker"] == selected_ticker, "Stock short name*"]
        if not match.empty:
            st.caption(f"Selected ETF: **{selected_ticker}** - {match.iloc[0]}")

    price_df = load_one_year_prices(selected_ticker)
    if price_df.empty:
        st.warning(f"No 1-year price series found for `{selected_ticker}`.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_df["Date"], y=price_df["Close"], mode="lines", name=selected_ticker))
        fig.update_layout(
            title=f"{selected_ticker} - Last 1 Year Price Trend",
            xaxis_title="Date",
            yaxis_title="Close",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_traces(line=dict(color="#7bf7d8", width=2.3))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 10 Holdings")
    holdings = load_top_holdings(selected_ticker)
    if holdings.empty:
        st.info(f"No top holdings file found for `{selected_ticker}` under `data/etf/holdings/top10`.")
    else:
        if "Weight (%)" in holdings.columns:
            holdings["Weight (%)"] = holdings["Weight (%)"].map(_format_pct)
        st.dataframe(holdings, use_container_width=True, hide_index=True)


def render_chatbot(tickers: list[str], metadata: pd.DataFrame) -> None:
    st.subheader("AI ETF Chatbot 🤖")
    if not tickers:
        st.info("No ticker available for chatbot context.")
        return

    current = st.session_state.get("selected_ticker", tickers[0])
    if current not in tickers or current == "0000":
        current = tickers[0]
    st.session_state.selected_ticker = current

    mode_col1, mode_col2 = st.columns([1.6, 2.4])
    with mode_col1:
        chat_mode = st.radio(
            "Ticker mode",
            options=["Select ETF", "Auto-detect from query/news"],
            horizontal=True,
            key="chat_ticker_mode",
        )
    with mode_col2:
        if chat_mode == "Select ETF":
            chat_selected_ticker = st.selectbox(
                "Chatbot ETF context",
                options=tickers,
                index=tickers.index(current),
                key="chat_selected_ticker",
            )
            st.session_state.selected_ticker = chat_selected_ticker
        else:
            chat_selected_ticker = None
            st.caption("Ticker will be inferred from your question when possible.")

    adv1, adv2, adv3, adv4 = st.columns([1, 2, 1.1, 1.1])
    with adv1:
        backend = st.selectbox("Backend", options=["transformers", "ollama", "vllm"], key="chat_backend")
    with adv2:
        model_name = st.text_input("Model Name", value=DEFAULT_LOCAL_QWEN_MODEL, key="chat_model")
    with adv3:
        use_response_cache = st.checkbox(
            "Use response cache",
            value=False,
            help="Turn on for faster repeat answers. Turn off for fresh generation (recommended).",
            key="chat_use_response_cache",
        )
    with adv4:
        auto_init_model = st.checkbox(
            "Auto-init model",
            value=True,
            help="Warm up model once so first message doesn't hang.",
            key="chat_auto_init_model",
        )

    if chat_mode == "Select ETF":
        st.caption(f"Chat context ticker: `{chat_selected_ticker}`")
    else:
        st.caption("Chat context ticker: `Auto` (inferred per query)")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "model_warmup_keys" not in st.session_state:
        st.session_state.model_warmup_keys = set()

    ctrl1, ctrl2 = st.columns([1.1, 3.0])
    with ctrl1:
        if st.button("Clear chat", key="clear_chat_btn"):
            st.session_state.chat_messages = []
            st.rerun()
    with ctrl2:
        st.caption("Tip: Clear chat to start a fresh conversation context.")

    if auto_init_model and backend == "transformers":
        try:
            _warmup_model_once(
                backend=backend,
                model_name=model_name,
                use_response_cache=use_response_cache,
            )
            st.caption("Model warm-up completed.")
        except Exception as exc:
            st.warning(f"Model warm-up failed: {exc}")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("evidence"):
                with st.expander("Evidence"):
                    st.json(msg["evidence"])

    prompt = st.chat_input("Ask about risks, diversification, alternatives, or outlook...")
    if not prompt:
        return

    user_msg = {"role": "user", "content": prompt, "evidence": None}
    st.session_state.chat_messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        try:
            resolved_ticker = chat_selected_ticker
            resolution_note = None
            if chat_mode != "Select ETF":
                if _is_recommendation_query(prompt):
                    ranked = rank_etf_candidates_hybrid(prompt, metadata, tickers, max_items=5)
                    if ranked:
                        top = ranked[0]
                        top2 = ranked[1] if len(ranked) > 1 else None
                        margin = float(top["final_score"]) - float(top2["final_score"]) if top2 else float(top["final_score"])
                        st.caption(
                            "Auto-detect recommendation mode. "
                            f"Top match `{top['ticker']}` score={float(top['final_score']):.2f}, margin={margin:.2f}"
                        )
                    response = build_recommendation_message(prompt, ranked, metadata)
                    evidence = {
                        "mode": "auto_recommendation",
                        "ranked_candidates": ranked[:3],
                    }
                    st.markdown(response)
                    assistant_msg = {"role": "assistant", "content": response, "evidence": evidence}
                    st.session_state.chat_messages.append(assistant_msg)
                    return

                resolved_ticker, resolution_note = infer_ticker_from_query(prompt, metadata, tickers)
                if resolved_ticker is None:
                    ranked = rank_etf_candidates_hybrid(prompt, metadata, tickers, max_items=3)
                    if ranked and float(ranked[0]["final_score"]) >= 0.16:
                        resolved_ticker = str(ranked[0]["ticker"])
                        candidate_text = ", ".join(
                            [f"`{x['ticker']}` ({float(x['final_score']):.2f})" for x in ranked]
                        )
                        st.caption(
                            "Auto-detected with moderate confidence from hybrid matching. "
                            f"Candidates: {candidate_text}"
                        )
                        resolution_note = "Auto-detected from hybrid matching (moderate confidence)."
                    else:
                        candidates = suggest_etf_candidates_from_query(prompt, metadata, max_items=3)
                        st.warning(
                            "Could not infer a confident single ticker from this query. "
                            "Tip: include a 4-digit ticker (e.g. 2800) or switch to Select ETF mode."
                        )
                        if candidates:
                            lines = "\n".join([f"- `{code}` {name}" for code, name in candidates])
                            response = (
                                "I couldn't confidently map your query to one ETF. "
                                "You can pick one of these likely candidates and ask again:\n"
                                f"{lines}"
                            )
                        else:
                            response = (
                                "I couldn't confidently map your query to one ETF. "
                                "Please include a 4-digit ticker or use Select ETF mode."
                            )
                        st.markdown(response)
                        assistant_msg = {"role": "assistant", "content": response, "evidence": {"ticker_inference": "no_confident_match"}}
                        st.session_state.chat_messages.append(assistant_msg)
                        return
                st.caption(f"Resolved ticker: `{resolved_ticker}`. {resolution_note}")
                st.markdown(build_etf_high_level_summary(resolved_ticker, metadata))

            result = generate_synthesis(
                ticker=resolved_ticker,
                user_query=prompt,
                backend=backend,
                qwen_model=model_name,
                enable_response_cache=use_response_cache,
            )

            # Retry once if we received timeout fallback text from model path.
            response = str(result.get("response", "")).strip()
            if backend == "transformers" and _is_timeout_fallback_response(response):
                st.caption("Retrying once to improve response quality...")
                retry_result = generate_synthesis(
                    ticker=resolved_ticker,
                    user_query=prompt,
                    backend=backend,
                    qwen_model=model_name,
                    enable_response_cache=False,
                )
                retry_response = str(retry_result.get("response", "")).strip()
                if retry_response and not _is_timeout_fallback_response(retry_response):
                    result = retry_result

            response = str(result.get("response", "")).strip() or "No response returned by synthesis engine."
            response = _fix_invalid_ticker_claim(response, resolved_ticker, tickers, metadata)
            response = _sanitize_resolved_ticker_response(response, resolved_ticker, tickers, metadata)
            evidence = result.get("data_evidence", {})
            if isinstance(evidence, dict):
                evidence["ticker_claim_guardrail"] = "applied"
            elapsed = time.perf_counter() - start
            st.markdown(response)
            st.caption(f"Response time: {elapsed:.2f}s")
            if evidence.get("answer_mode"):
                st.caption(f"Answer mode: `{evidence.get('answer_mode')}`")
            if evidence:
                with st.expander("Evidence"):
                    st.json(evidence)
            assistant_msg = {"role": "assistant", "content": response, "evidence": evidence}
        except Exception as exc:
            error_text = (
                "Chatbot could not generate an answer. "
                f"Please verify local model/data dependencies. Error: `{exc}`"
            )
            st.error(error_text)
            assistant_msg = {"role": "assistant", "content": error_text, "evidence": {"error": str(exc)}}

    st.session_state.chat_messages.append(assistant_msg)


def main() -> None:
    st.set_page_config(
        page_title="HK ETF Financial Intelligence Platform",
        page_icon=":material/finance:",
        layout="wide",
    )

    apply_frontend_theme()
    metadata, metadata_path = load_metadata()
    tickers = discover_tickers()
    if not metadata.empty and not tickers:
        tickers = sorted(metadata["Ticker"].dropna().astype(str).tolist())

    render_hero(tickers=tickers, metadata=metadata)
    st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (local)")

    if "selected_ticker" not in st.session_state and tickers:
        st.session_state.selected_ticker = tickers[0]
    if "model_warmup_keys" not in st.session_state:
        st.session_state.model_warmup_keys = set()

    # Warm up default chatbot model at app launch to reduce first-message hanging.
    try:
        _warmup_model_once(
            backend="transformers",
            model_name=DEFAULT_LOCAL_QWEN_MODEL,
            use_response_cache=False,
        )
    except Exception:
        # Non-fatal: chatbot tab will still attempt warmup again when opened.
        pass

    if metadata.empty:
        st.warning(
            "ETF metadata is currently unavailable. "
            f"Expected file: `{metadata_path}`. Screener and metadata labels may be limited."
        )

    tab_weekly, tab_screener, tab_explorer, tab_chat = st.tabs(
        ["⚡ 1W Winners/Losers", "🔎 ETF Screener", "📈 ETF Explorer", "🤖 AI Chatbot"]
    )

    with tab_weekly:
        render_weekly_winners_losers(tickers, metadata)
    with tab_screener:
        render_screener(metadata)
    with tab_explorer:
        render_explorer(tickers, metadata)
    with tab_chat:
        render_chatbot(tickers, metadata)

    with st.expander("Data and model paths", expanded=False):
        st.write(
            {
                "metadata_expected": metadata_path,
                "ohlcv_root": str(_default_etf_root() / "ohlcv"),
                "holdings_root": str(_default_etf_root() / "holdings" / "top10"),
                "synthesis_dna_root": str(PROJECT_ROOT / "model_output" / "dna"),
                "synthesis_synapse_root": str(PROJECT_ROOT / "model_output" / "synpse"),
            }
        )


if __name__ == "__main__":
    main()
