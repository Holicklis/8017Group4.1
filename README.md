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
  - Produces `data/etf/processed/financial_dna.parquet`.
- **Clustering module**: `src/model/dna/model_core.py`
  - Standardization: `StandardScaler`.
  - Dimensionality reduction: `PCA` with configurable variance threshold.
  - Clustering: `KMeans` with silhouette-based auto-k selection logic.
  - Outputs perspective-level cluster artifacts to `data/etf/processed/cluster_views/`.
- **Advisory layer**: `src/model/dna/advisory_logic.py`
  - Converts cluster geometry into actionable candidate sets.
  - Writes outputs to `data/etf/processed/advisory/`.

**Value Add**  
Detects false diversification, highlights mathematically similar alternatives, and supports lower-cost global substitution analysis.

### 🧠 Model 2: Synapse (Semantic Connector) — **In Progress**
**Purpose**  
Bridge ETF risk disclosures with real-world narratives and event flow.

**Methodology (Target Design)**  
- Parse HKEX prospectus/key-facts text into sentence-level corpus.
- Encode text with Sentence-BERT embeddings.
- Match news/risk statements by cosine similarity and semantic re-ranking.

**Current Status**  
- Foundation code exists in `src/model/synapse/model.py`.
- Data pipeline support exists through document scraping + PDF text extraction.
- Production-grade scoring, calibration, and evaluation are planned next.

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
│       └── processed/
│           ├── financial_dna.parquet
│           ├── cluster_views/
│           └── advisory/
├── src/
│   ├── etf_pipeline.py
│   ├── data_ingestion/
│   │   ├── etf_market_data_fetcher.py
│   │   └── etf_top_holdings_data_fetcher.py
│   ├── hkex_etf/
│   │   ├── etf_metadata_export.py
│   │   └── etf_document_scraper.py
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

## 📥 Notebook / chatbot datasets (optional)

**Option A — Automated (recommended)**  
From the repo root, with a [Kaggle API token](https://www.kaggle.com/settings) (**recommended:** `export KAGGLE_API_TOKEN=KGAT_…`, or legacy `kaggle.json` in the project root / `~/.kaggle/`):

```bash
export KAGGLE_API_TOKEN=KGAT_your_token_here   # optional if kaggle.json is present
uv run python scripts/download_data.py
```

This pulls complaints, phrasebank, and Q&A from public Hugging Face mirrors, then downloads alpha-insights, finance survey, S&P 500 prices, financial news, and FinSen from Kaggle. If you have no Kaggle token yet, the script still downloads the first three sources; run it again after adding `kaggle.json`.

**Option B — OneDrive**  
When raw artifacts are too large for GitHub (~3.3 GB raw), data can be shared via OneDrive. Copy the entire `data/` folder into the project root so the tree matches the `data/` section under **Project Structure** above.

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
1. Engineer feature matrix (`financial_dna.parquet`).
2. Run PCA + K-Means by perspective.
3. Generate advisory artifacts (home-bias and hidden-twin candidates).

### Step 4: Synapse Intelligence
1. Embed documentation text.
2. Match external narratives/headlines via cosine similarity.
3. Produce risk-alignment signals.

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

### 4) Run Data Fetchers Individually
```bash
uv run python src/data_ingestion/etf_market_data_fetcher.py
uv run python src/data_ingestion/etf_top_holdings_data_fetcher.py
```

## 🎓 Academic Foundation
This project is developed in an HKU academic context and is grounded in quantitative portfolio research.  
The clustering design is inspired by **Agarwal et al. (2017)** and related literature on PCA-driven stock portfolio structuring, adapted here for ETF-level diversification and home-bias mitigation in the Hong Kong market.

## ⚠️ Disclaimer
This repository is for academic and research purposes only.  
It does not constitute investment advice, solicitation, or a recommendation to buy/sell any security.
