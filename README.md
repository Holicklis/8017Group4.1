# HK ETF Intelligence Platform
### Evidence-Driven Global Diversification Beyond Home Bias

## 🌍 Introduction
Hong Kong retail portfolios often exhibit **home bias**: concentrated exposure to familiar local markets despite the risk-reduction potential of global allocation.  
**HK ETF Intelligence Platform** addresses this gap with an evidence-based workflow that combines market microstructure, fund fundamentals, prospectus text, and semantic risk signals.

The platform is built as a three-model ecosystem:
- **Financial DNA** for quantitative structure discovery,
- **Synapse** for semantic risk/event linkage,
- **Financial Synthesis** for human-centric explanation and decision support.

## 🧩 The Model Ecosystem

### 🧬 Model 1: Financial DNA (Mathematical Core) — **Implemented**
**Purpose**  
Identify hidden structural relationships between HK ETFs using empirical behavior, not marketing labels.

**Methodology**  
- Build a unified feature table from metadata, OHLCV, market-cap, and holdings-derived signals.
- Standardize features and run **PCA** to retain high explanatory variance while reducing noise.
- Apply **K-Means** clustering across multiple perspectives (for example `return_risk_profile`, `macro_sensitivity`).
- Use perspective-level clusters to generate advisory candidates (for example home-bias mitigation and hidden twins).

**Technical Details (Current Implementation)**  
- **Data engineering module**: `src/model/dna/data_engine.py`
  - Loads ETF universe and harmonizes ticker-level data.
  - Engineers return, volatility, yield, concentration, and cross-asset relationship features.
  - Produces `model_output/dna/financial_dna.parquet`.
- **Clustering module**: `src/model/dna/model_core.py`
  - Standardization: `StandardScaler`.
  - Dimensionality reduction: `PCA` with configurable variance threshold.
  - Clustering: `KMeans` with silhouette-based auto-k selection logic.
  - Outputs perspective-level cluster artifacts to `model_output/dna/cluster_views/`.
- **Advisory layer**: `src/model/dna/advisory_logic.py`
  - Converts cluster geometry into actionable candidate sets.
  - Writes outputs to `model_output/dna/advisory/`.

**Value Add**  
Detects false diversification, highlights mathematically similar alternatives, and supports lower-cost global substitution analysis.

### 🧠 Model 2: Synapse (Semantic Connector) — **Implemented**
**Purpose**  
Bridge ETF risk disclosures with real-world narratives and event flow.

**Motivation (Retail Investor Use Case)**  
- Retail investors usually understand stories ("rates rising", "China policy shift") but not always the **ETF-level risk transmission** of those stories.
- Most public sentiment datasets are **single-stock focused**, while ETF risk is often **macro and cross-asset**.
- Synapse maps news narratives to ETF risk profiles so users can discover both:
  - risk concentration they may already hold, and
  - opportunity candidates aligned with their optimism/worry scenario.

**Methodology (Current Design)**  
- Build one ETF-level semantic profile from prospectus/key-facts texts:
  - section-aware filtering,
  - noise/boilerplate removal,
  - near-duplicate suppression,
  - strict sentence/character budgets.
- Encode profile corpus with local bi-encoder embeddings.
- Retrieve top candidates by semantic similarity.
- Apply optional metadata/tag boosting and optional lightweight reranking.
- Add financial sentiment signal (FinBERT) to produce combined relevance+sentiment views.

**Technical Details (Text Processing Focus)**  
- Source extraction: `src/text_extraction/pdf_text_extractor.py`
  - PDF cleanup and sentence normalization
  - keyword-based risk/component tagging
  - per-ETF profile generation (`model_output/synpse/etf_profiles.csv`)
- Retrieval core: `src/model/synapse/model.py`
  - profile-level corpus (one row per ticker)
  - cache-versioned embeddings
  - fast/quality model presets
- Batch inference + visualization: `src/model/synapse/run_news_events.py`
  - run on full news CSVs
  - export top-k matches + score diagnostics + visual artifacts
- Stability testing: `src/model/synapse/semantic_clustering_stability.py`
  - strict paraphrase stress test (`100 concepts x 20 rewrites` default)
  - top-k overlap + tie-aware stability + score-correlation diagnostics
  - optional light reranking and query canonicalization for robustness checks

**No External API Principle**  
- Synapse avoids external API dependency for scoring/tagging:
  - faster and cheaper at scale,
  - easier low-latency reaction for new headlines,
  - deterministic and reproducible local runs,
  - can use internal ETF profile context that generic external APIs do not have.

**Value Add**  
Generates early risk-alignment signals when external narratives resemble an ETF's documented risk DNA.

### 🤖 Model 3: Financial Synthesis (Intelligent Interface) — **Planned**
**Purpose**  
Translate quantitative clusters and semantic signals into investor-readable action guidance.

**Methodology (Target Design)**  
- Retrieval-augmented response layer over Financial DNA + Synapse outputs.
- Explainable recommendation templates (why switch, where risk overlaps, what diversification improves).
- User-facing conversational interface for decision support.

**Current Status**  
- Architecture defined; implementation will be added after Synapse stabilization.

**Value Add**  
Delivers a personalized "Global Navigator" experience that converts technical analytics into practical portfolio actions.

## ✨ Key Features
- **ETF Screener**: Quantitative ETF profiling from metadata, OHLCV, market cap, and holdings signals.
- **Diversification Analysis**: Cluster-based similarity mapping, hidden twins, and home-bias candidate detection.
- **AI Chat/Advisor Layer**: Human-readable interpretation of statistical outputs and risk context.

## 🛠 Tech Stack
- **Python**
- **Scikit-Learn** (PCA, K-Means, clustering diagnostics)
- **Sentence-Transformers** (semantic embeddings for Synapse workflows)
- **PyMuPDF / PDF pipeline utilities** (document extraction flow)
- **yfinance** (market and fund holdings data ingestion)
- **Streamlit / Dash** (optional UI serving layer for interactive exploration)

## 🧱 Project Structure
```text
8017Group4.1/
├── data/
│   └── etf/
│       ├── instruments/
│       │   └── all_hk_etf.csv
│       ├── summary/
│       │   └── ETP_Data_Export.xlsx
│       ├── ohlcv/
│       ├── market_cap/
│       ├── holdings/
│       │   └── top10/
│       ├── documentation/
│       │   └── <ticker>/{pdf,csv}/
├── model_output/
│   ├── dna/
│   │   ├── financial_dna.parquet
│   │   ├── cluster_views/
│   │   └── advisory/
│   ├── synpse/
│   │   ├── etf_profiles.csv
│   │   ├── cache/
│   │   ├── benchmark_side_by_side_*.json
│   │   ├── news_events_run_*/
│   │   └── stability_run_*/
│   └── Synthesis/
├── src/
│   ├── etf_pipeline.py
│   ├── data_ingestion/
│   │   └── provider/
│   │       ├── hkex/
│   │       │   └── hkex_etf/
│   │       │       ├── etf_metadata_export.py
│   │       │       └── etf_document_scraper.py
│   │       └── yfinance/
│   │           ├── etf_market_data_fetcher.py
│   │           └── etf_top_holdings_data_fetcher.py
│   ├── text_extraction/
│   │   └── pdf_text_extractor.py
│   └── model/
│       ├── dna/
│       │   ├── data_engine.py
│       │   ├── model_core.py
│       │   ├── advisory_logic.py
│       │   └── visualize_clusters.py
│       └── synapse/
│           └── model.py
├── pyproject.toml
└── README.md
```

## 🔄 Workflow & Pipeline

### Step 1: Data Ingestion
1. Export HKEX ETF universe metadata.
2. Download OHLCV + market cap from market APIs.
3. Fetch top holdings data.
4. Scrape ETF documents (prospectus / key facts).

### Step 2: Text Processing
1. Convert PDFs to clean sentence-level CSV.
2. Build corpus for semantic indexing and retrieval.

### Step 3: Financial DNA Modeling
1. Engineer feature matrix (`model_output/dna/financial_dna.parquet`).
2. Run PCA + K-Means by perspective.
3. Generate advisory artifacts (home-bias and hidden-twin candidates).

### Step 4: Synapse Intelligence
1. Build ETF profile corpus in `model_output/synpse/`.
2. Embed documentation text and retrieve by semantic relevance.
3. Optionally add sentiment and stability diagnostics.

### Step 5: Financial Synthesis
1. Merge quantitative and semantic evidence.
2. Present investor-facing insights for global diversification decisions.

## 🚀 Installation & Usage

### 1) Environment Setup
```bash
# install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# install project dependencies
uv sync
```

This repository follows a modern `src/` layout with explicit package boundaries
via `__init__.py` files under `src/`.

### 2) Run Core Data Pipeline
```bash
uv run python src/etf_pipeline.py
```

Useful options:
```bash
uv run python src/etf_pipeline.py --skip-export
uv run python src/etf_pipeline.py --skip-documents
uv run python src/etf_pipeline.py --skip-text-extract
uv run python src/etf_pipeline.py --summary-file "data/etf/summary/ETP_Data_Export.xlsx"
uv run python src/etf_pipeline.py --no-headless
```

### 3) Run Financial DNA Modules
```bash
uv run python src/model/dna/data_engine.py
uv run python src/model/dna/model_core.py
uv run python src/model/dna/advisory_logic.py
uv run python src/model/dna/visualize_clusters.py
```

Run the full DNA pipeline in one command:
```bash
uv run python src/model/dna/run.py
```

### 4) Run Data Fetchers Individually
```bash
uv run python src/data_ingestion/provider/yfinance/etf_market_data_fetcher.py
uv run python src/data_ingestion/provider/yfinance/etf_top_holdings_data_fetcher.py
```

### 5) Synapse Usage Guide
```bash
# Build / refresh ETF profiles from document corpus
uv run python src/text_extraction/pdf_text_extractor.py

# Quick query test (profile corpus, fast preset)
uv run python src/model/synapse/model.py --query "Fed pause supports duration-sensitive assets" --top-k 5

# Side-by-side benchmark (profile vs sentence corpus)
uv run python src/model/synapse/evaluate_benchmark.py --preset fast --top-k 3 --sentence-row-cap 4000

# Full news run with financial sentiment integration
uv run python src/model/synapse/run_news_events.py \
  --input-csv "data/news/financial_news_events.csv" \
  --preset fast \
  --corpus-mode profile \
  --top-k 3 \
  --sentiment-model "ProsusAI/finbert" \
  --sentiment-weight 0.25

# Semantic stability stress test
uv run python src/model/synapse/semantic_clustering_stability.py \
  --preset fast \
  --corpus-mode profile \
  --top-k 5 \
  --num-concepts 100 \
  --variants-per-concept 20 \
  --tie-epsilon 0.01 \
  --enable-light-rerank \
  --cross-top-n 12
```

Run the full Synapse pipeline (benchmark + news + stability) in one command:
```bash
uv run python src/model/synapse/run.py
```

Run the Synthesis model entrypoint:
```bash
uv run python src/model/synthesis/run.py
```

Default model outputs now go to:
- `model_output/dna/...`
- `model_output/synpse/...`
- `model_output/Synthesis/...`

## 🎓 Academic Foundation
This project is developed in an HKU academic context and is grounded in quantitative portfolio research.  
The clustering design is inspired by **Agarwal et al. (2017)** and related literature on PCA-driven stock portfolio structuring, adapted here for ETF-level diversification and home-bias mitigation in the Hong Kong market.

## ⚠️ Disclaimer
This repository is for academic and research purposes only.  
It does not constitute investment advice, solicitation, or a recommendation to buy/sell any security.