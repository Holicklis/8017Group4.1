# 🧩 Model Details

This document keeps the full technical description of the three-model system:

- 🧬 **Financial DNA** — quantitative structure discovery
- 🔗 **Synapse** — semantic risk-event linkage
- ✨ **Financial Synthesis** — investor-facing response generation

It is intentionally detailed and complements the beginner-friendly `README.md`.

## 1) 🧬 Financial DNA (Mathematical Core)

### 🎯 Purpose
Identify hidden structural relationships between HK ETFs using empirical behavior instead of only marketing labels.

### 🔬 Method
- Build unified ETF feature table from metadata, OHLCV, market-cap, and holdings-derived signals.
- Standardize features and apply `PCA` for dimensionality reduction.
- Apply `KMeans` clustering across multiple perspectives (for example `return_risk_profile`, `macro_sensitivity`).
- Convert cluster geometry into actionable advisory candidates.

### ⚙️ Implementation
- **Data engineering:** `src/model/dna/data_engine.py`
  - Loads ETF universe and harmonizes ticker-level datasets.
  - Engineers return, volatility, yield, concentration, and cross-asset relationship features.
  - Produces `model_output/dna/financial_dna.parquet`.
- **Clustering core:** `src/model/dna/model_core.py`
  - `StandardScaler`, `PCA`, `KMeans`.
  - Supports silhouette-based auto-k logic.
  - Writes perspective-level artifacts to `model_output/dna/cluster_views/`.
- **Advisory logic:** `src/model/dna/advisory_logic.py`
  - Generates home-bias and hidden-twin candidate outputs.
  - Writes to `model_output/dna/advisory/`.

### 💡 Value
- Detects false diversification.
- Highlights mathematically similar alternatives.
- Supports lower-cost substitution and cross-market diversification analysis.

## 2) 🔗 Synapse (Semantic Connector)

### 🎯 Purpose
Bridge ETF risk disclosures with real-world narratives and event flow.

### 💬 Why it matters for retail investors
- Investors often understand stories (`rates`, `policy`, `macro shocks`) more easily than factor-level risk transmission.
- ETF risk is frequently macro and cross-asset, not single-stock only.
- Synapse maps narratives to ETF profiles to show both:
  - potential risk concentration in current holdings
  - scenario-aligned opportunity candidates

### 🔬 Method
- Build one ETF-level semantic profile from prospectus and key-facts text.
- Use section-aware filtering, noise/boilerplate reduction, near-duplicate suppression, and budget controls.
- Encode profiles with local sentence-embedding models.
- Retrieve top semantic matches and optionally apply tag boosting / lightweight reranking.
- Optionally combine relevance with financial sentiment signal.

### ⚙️ Implementation
- **Text extraction pipeline:** `src/text_extraction/pdf_text_extractor.py`
  - PDF cleanup and sentence normalization.
  - Risk/component tagging.
  - Writes profile corpus (for example `model_output/synpse/etf_profiles.csv`).
- **Retrieval core:** `src/model/synapse/model.py`
  - Profile-level corpus retrieval.
  - Cache-versioned embeddings.
  - Fast/quality model presets.
- **Batch event runs:** `src/model/synapse/run_news_events.py`
  - Runs top-k matching over financial news events.
  - Exports diagnostics and visual artifacts.
- **Stability testing:** `src/model/synapse/semantic_clustering_stability.py`
  - Paraphrase robustness stress tests.
  - Overlap/tie-aware stability and score-correlation diagnostics.

### 🏠 Local-first principle
Synapse is designed for local reproducible runs (cost, latency, privacy, reproducibility).

## 3) ✨ Financial Synthesis (Intelligent Interface)

### 🎯 Purpose
Translate quantitative and semantic signals into investor-readable guidance with explicit evidence.

### 🔄 Core behavior
- Pulls DNA context:
  - dominant cluster id
  - perspective-level mapping
  - top alternatives from advisory outputs
- Pulls Synapse alerts:
  - threshold-based filtering
  - optional query-level reranking
- Routes output language (English/Chinese detection).
- Produces structured response:
  - natural-language answer
  - `data_evidence` block (cluster + similarity evidence)

### ⚙️ Implementation
- Synthesis engine: `src/model/synthesis/synthesis_engine.py`
- CLI entrypoint: `src/model/synthesis/run.py`
- **Public integration interface:**
  - `generate_synthesis(...)`
  - `get_synthesis_context(...)`
  - `get_synapse_alerts(...)`

### 🖥️ Backends
- `transformers` (local in-process model)
- `ollama` (local HTTP endpoint)
- `vllm` (local HTTP endpoint)

### ⏱️ Caching / runtime notes
- In standard app flow, the default engine instance is reused in process.
- First local `transformers` response loads model/tokenizer.
- Later responses in the same process reuse loaded objects.

## 4) 📝 QnA Generation and Fine-Tuning Data

### 📚 Source corpus
Prospectus and product key facts sentence-level CSVs:
- `data/etf/documentation/<ticker>/csv`

### 🛠️ Generation module
- `src/model/synthesis/generate_finetune_qa.py`
- Tasks:
  - sentence cleanup and filtering
  - extraction of key facts (fees, index, manager, lot size, policy signals)
  - generation of topic-grounded QnA pairs
  - quality gates and deduplication

### 📤 Outputs
- Flat QA: CSV/JSONL
- Chat training format: ChatML JSONL
- Typical output root: `model_output/Synthesis/finetune_all/`

### 🎓 Fine-tuning
- Script: `src/model/synthesis/finetune_qwen.py`
- Typical base model: `Qwen/Qwen2.5-7B-Instruct`

## 5) 🔁 End-to-End Data Flow

1. Ingestion updates metadata, OHLCV, holdings, and documentation.
2. DNA builds quantitative features and advisory artifacts.
3. Synapse builds semantic profiles and event similarity outputs.
4. Synthesis combines DNA + Synapse and returns advisor-style responses.
5. Streamlit app surfaces outputs for screening, exploration, and Q&A.

## 6) 📁 Key Output Paths

- **DNA:**
  - `model_output/dna/financial_dna.parquet`
  - `model_output/dna/cluster_views/`
  - `model_output/dna/advisory/`
- **Synapse:**
  - `model_output/synpse/etf_profiles.csv`
  - `model_output/synpse/news_events_run_*/news_event_topk_matches.csv`
- **Synthesis / QnA:**
  - `model_output/Synthesis/finetune_all/`

