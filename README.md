# 🇭🇰 HK ETF Intelligence Platform
Evidence-driven ETF analytics and AI insights for Hong Kong retail investors.

## 📋 Overview
This project helps investors explore HK ETFs with three connected intelligence modules:

- 🧬 **Financial DNA** — quantitative ETF structure and clustering
- 🔗 **Synapse** — semantic link between ETF risk text and market events
- ✨ **Financial Synthesis** — multilingual advisor-style response with evidence

The repository also includes a Streamlit application for screening ETFs, tracking weekly winners/losers, exploring holdings, and chatting with the synthesis model.

## 📖 Model Documentation
Detailed model write-up is preserved in:
- 📄 `MODEL_DETAILS.md` (repo root)

## ⭐ Key Features

### 🖥️ Platform Features
- Build ETF feature matrix from metadata, OHLCV, holdings, and market-cap signals
- Cluster ETFs with PCA + K-Means for diversification and similarity discovery
- Generate advisory outputs (home-bias alternatives and hidden twins)
- Match ETF semantic risk profiles to financial news/events
- Generate synthesis responses with traceable evidence (`cluster context` + `news similarity`)
- Generate and fine-tune QnA datasets for local Qwen models

### 📱 App Features (`app/hk_etf_intelligence_app.py`)
- 📈 **1W Winners/Losers:** top/bottom 1-week ETF movers, including category splits
- 🔍 **ETF Screener:** filter ETFs using HKEX metadata fields
- 🗺️ **ETF Explorer:** 1-year price chart + top holdings (symbol, name, weight)
- 🤖 **AI Chatbot:** synthesis Q&A with evidence block

## 🚀 Quick Start (Beginner Friendly)

### 1) 📦 Install dependencies
```bash
# install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# from repo root
uv sync
```

**Git LFS (large weights / checkpoints):** Install Git LFS once per machine (`brew install git-lfs` on macOS), then run `git lfs install` in this repo. After cloning, run `git lfs pull` so `*.pt`, `*.safetensors`, and `*.joblib` pointer files download real binaries. **`corpus_embeddings*.pt` files are not in the repo** (they are rebuilt locally). GitHub enforces [LFS bandwidth and storage quotas](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

### 2) ▶️ Launch the app
```bash
uv run streamlit run app/hk_etf_intelligence_app.py
```

### 3) 📂 Minimum data required for full app experience
- `data/etf/summary/ETP_Data_Export.xlsx`
- `data/etf/ohlcv/<ticker>/ohlcv.parquet`
- `data/etf/holdings/top10/<ticker>/top_holdings.parquet`
- `model_output/dna/cluster_views/cluster_perspectives.parquet`
- `model_output/dna/advisory/home_bias_candidates.parquet`
- `model_output/dna/advisory/hidden_twin_candidates.parquet`
- `model_output/synpse/news_events_run_*/news_event_topk_matches.csv`

> 💡 **Note:** If your local repo uses `data/ETF/` (uppercase) instead of `data/etf/`, the app automatically falls back.

## 📘 App Usage Guide

### 🗂️ Tabs
- **1W Winners/Losers:**
  - overall top and bottom weekly performers
  - top/bottom 5 by category (`Equity`, `Commodity`, `Fixed Income`)
- **ETF Screener:**
  - search by ticker/name
  - filter by category and numeric ranges
  - push selected ETF into Explorer and Chatbot context
- **ETF Explorer:**
  - 1-year close-price trend
  - top holdings table with symbol + name + weight
- **AI Chatbot:**
  - backend options: `transformers`, `ollama`, `vllm`
  - returns response + evidence

### 🔌 Chatbot backends
- **`transformers`** (default): local in-process model
- **`ollama`:** requires local Ollama endpoint (`http://localhost:11434`)
- **`vllm`:** requires local vLLM endpoint (`http://localhost:8000`)

For `transformers`, first response includes model load time; later responses reuse loaded model in the same app process.

## 🛠️ Full Pipeline Commands

### 🧬 Core ETF pipeline
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

### 🧬 Financial DNA
```bash
uv run python src/model/dna/data_engine.py
uv run python src/model/dna/model_core.py
uv run python src/model/dna/advisory_logic.py
uv run python src/model/dna/visualize_clusters.py
```

Run all DNA steps:
```bash
uv run python src/model/dna/run.py
```

### 📡 Data fetchers
```bash
uv run python src/data_ingestion/provider/yfinance/etf_market_data_fetcher.py
uv run python src/data_ingestion/provider/yfinance/etf_top_holdings_data_fetcher.py
```

### 🔗 Synapse
```bash
# build ETF profiles
uv run python src/text_extraction/pdf_text_extractor.py

# quick query test
uv run python src/model/synapse/model.py --query "Fed pause supports duration-sensitive assets" --top-k 5

# run full synapse pipeline
uv run python src/model/synapse/run.py
```

### ✨ Synthesis
```bash
# synthesis CLI entrypoint
uv run python src/model/synthesis/run.py
```

### 📝 QnA generation and fine-tuning
```bash
# generate QnA for one ticker
uv run python src/model/synthesis/generate_finetune_qa.py \
  --csv-dir "data/etf/documentation/02800/csv" \
  --output-dir "model_output/Synthesis/finetune" \
  --ticker 02800 \
  --max-pairs 240

# generate QnA for all tickers
uv run python src/model/synthesis/generate_finetune_qa.py \
  --all-csv \
  --documentation-root "data/etf/documentation" \
  --output-dir "model_output/Synthesis/finetune_all" \
  --max-pairs 40
```

Fine-tune Qwen:
```bash
uv run python src/model/synthesis/finetune_qwen.py \
  --dataset-path "model_output/Synthesis/finetune_all/all_tickers_finetune_qa_chatml.jsonl" \
  --output-dir "model_output/Synthesis/finetuned/qwen2.5_7b_lora_full" \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-length 1024
```

## 🗂️ Project Structure
```text
8017Group4.1/
├── app/
│   ├── chatbot_app.py
│   └── hk_etf_intelligence_app.py
├── MODEL_DETAILS.md
├── data/
│   └── etf/
│       ├── instruments/
│       ├── summary/
│       ├── ohlcv/
│       ├── market_cap/
│       ├── holdings/top10/
│       └── documentation/<ticker>/{pdf,csv}/
├── model_output/
│   ├── dna/
│   ├── synpse/
│   └── Synthesis/
│       └── finetune_all/
├── src/
│   ├── etf_pipeline.py
│   ├── data_ingestion/provider/
│   ├── text_extraction/
│   └── model/
│       ├── dna/
│       ├── synapse/
│       └── synthesis/
├── pyproject.toml
└── README.md
```

## ❓ App FAQ

### 1) App starts but some tabs are empty
Most likely some local files are missing. Check:
- `ETP_Data_Export.xlsx` for screener
- `ohlcv.parquet` for price and winners/losers
- `top_holdings.parquet` for holdings table
- DNA/Synapse outputs for chatbot evidence

### 2) Chatbot is slow on first response
This is expected for `transformers` backend because model weights load on first call.
Later calls are faster in the same process.

### 3) Why does chatbot fail with backend errors?
- `ollama` needs local service on `localhost:11434`
- `vllm` needs local service on `localhost:8000`
- `transformers` needs local compute/memory and model access

### 4) Why do I see `data/ETF` vs `data/etf` path confusion?
The app handles both automatically. If one folder is missing, it falls back to the other.

### 5) Why do holdings sometimes show limited columns?
Holdings schemas may vary by source/version. The app tries to map columns robustly and display `Symbol`, `Name`, and `Weight (%)` when available.

### 6) How do I quickly test that synthesis data exists?
```bash
uv run python src/model/synthesis/run.py --ticker 2800 --query "What are key risks?"
```

## 🧰 Tech Stack
- Python 3.10+
- pandas, numpy, scikit-learn
- sentence-transformers, transformers, torch
- streamlit, plotly
- yfinance

## 🎓 Academic Context
This project was developed in an HKU academic setting for ETF diversification research and investor-facing explainability.

## ⚠️ Disclaimer
This repository is for academic and research purposes only.  
It is **not** investment advice or a recommendation to buy/sell any security.
