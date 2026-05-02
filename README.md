# HK ETF Intelligence Platform

Evidence-driven ETF analytics and AI-style explanations for Hong Kong ETFs: clustering and advisory signals (**Financial DNA**), semantic links between disclosures and news (**Synapse**), and an optional advisor-style chat layer (**Financial Synthesis**). A **Streamlit** app ties screening, exploration, and chat into one UI.

---

## Table of contents

1. [Overview](#overview)
2. [What is in Git vs what you must produce locally](#what-is-in-git-vs-what-you-must-produce-locally)
3. [Requirements](#requirements)
4. [Clone and install](#clone-and-install)
5. [Data you need](#data-you-need)
6. [Fine-tune locally (size limits)](#fine-tune-locally-size-limits)
7. [Run the app](#run-the-app)
8. [Pipeline commands](#pipeline-commands)
9. [Project layout](#project-layout)
10. [Troubleshooting](#troubleshooting)
11. [Tech stack](#tech-stack)
12. [Disclaimer](#disclaimer)

---

## Overview

| Module | Role |
|--------|------|
| **Financial DNA** | Feature engineering, PCA + K-Means clustering, advisory outputs (e.g. home-bias and “hidden twin” candidates). |
| **Synapse** | ETF text profiles from prospectuses/key facts, embeddings, and similarity to news/events. |
| **Financial Synthesis** | Combines DNA + Synapse into readable answers with an evidence block; can use local Transformers, **Ollama**, or **vLLM**. |

Deeper theory and file-level references: **`MODEL_DETAILS.md`** (repository root).

---

## What is in Git vs what you must produce locally

| Artifact | In this repo? | Notes |
|----------|----------------|--------|
| Source code, `pyproject.toml`, app | Yes | — |
| **`data/`** (OHLCV, holdings, PDFs/CSVs, exports) | **No** (large; not committed) | Place files locally under `data/etf/…` (see [Data you need](#data-you-need)). |
| **`corpus_embeddings*.pt`** (Synapse caches) | **No** | Ignored; rebuilt when you run Synapse. |
| **Full fine-tune checkpoints** (LoRA, optimizer states, multi-GB dirs under `model_output/Synthesis/finetuned/…`) | **No** | **You must fine-tune locally** — see [Fine-tune locally](#fine-tune-locally-size-limits). |
| Smaller binaries tracked as **Git LFS** | Maybe (`*.pt`, `*.safetensors`, `*.joblib` per `.gitattributes`) | After clone: `git lfs install` and `git lfs pull`. [GitHub LFS has storage and bandwidth limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). |

**Takeaway:** cloning the repo gives you code and (if present via LFS) *some* weights. **Project-scale fine-tunes and embeddings are expected to be produced on your machine**, not downloaded fully from GitHub.

---

## Requirements

- **Python 3.10+** (see `pyproject.toml`).
- **[uv](https://docs.astral.sh/uv/)** for dependencies: `uv sync` from the repo root.
- **Disk:** several GB for data; **much more** if you fine-tune a 7B-class model (checkpoints, cache).
- **GPU:** strongly recommended for local **Transformers** chat and for **LoRA fine-tuning**; CPU may work for DNA/Synapse-only workflows or small models.
- **Git LFS:** required only if the repo tracks LFS objects you need; install once (`brew install git-lfs` on macOS), then `git lfs install`, and after each clone `git lfs pull`.

---

## Clone and install

```bash
git clone git@github.com:Holicklis/8017Group4.1.git
cd 8017Group4.1

# Optional: fetch Git LFS binaries if your clone uses them
git lfs install
git lfs pull

# Install Python dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync
```

---

## Data you need

The app and pipelines expect data under **`data/etf/`** (the code can fall back to **`data/ETF/`** if you already use that layout).

**Minimum for a full Streamlit experience** (from existing README paths):

- `data/etf/summary/ETP_Data_Export.xlsx`
- `data/etf/ohlcv/<ticker>/ohlcv.parquet`
- `data/etf/holdings/top10/<ticker>/top_holdings.parquet`
- `model_output/dna/cluster_views/cluster_perspectives.parquet`
- `model_output/dna/advisory/home_bias_candidates.parquet`
- `model_output/dna/advisory/hidden_twin_candidates.parquet`
- `model_output/synpse/news_events_run_*/news_event_topk_matches.csv`

Populate **`data/`** using your course or team instructions (e.g. HKEX exports, Kaggle bundles, or your own ingestion). Run the [pipelines](#pipeline-commands) to regenerate **`model_output/`** where needed.

---

## Fine-tune locally (size limits)

**Why local fine-tuning?**

- **GitHub blocks regular Git blobs over 100 MB.** Optimizer and full checkpoint folders are often hundreds of MB each.
- **Git LFS** is meant for large files but has **account storage and bandwidth quotas**; shipping multi–gigabyte fine-tune runs is usually impractical for a course repo.
- Therefore **LoRA / Qwen fine-tune artifacts under `model_output/Synthesis/finetuned/` are not treated as something everyone downloads from Git.** You **generate them on your machine** when you need that workflow.

**Recommended workflow**

1. Ensure prospectus/key-facts CSVs exist, e.g. `data/etf/documentation/<ticker>/csv` (see `MODEL_DETAILS.md`).
2. **Build QnA / ChatML data:**
   - One ticker example:

     ```bash
     uv run python src/model/synthesis/generate_finetune_qa.py \
       --csv-dir "data/etf/documentation/02800/csv" \
       --output-dir "model_output/Synthesis/finetune" \
       --ticker 02800 \
       --max-pairs 240
     ```

   - All tickers (adjust paths as needed):

     ```bash
     uv run python src/model/synthesis/generate_finetune_qa.py \
       --all-csv \
       --documentation-root "data/etf/documentation" \
       --output-dir "model_output/Synthesis/finetune_all" \
       --max-pairs 40
     ```

3. **Fine-tune** (example; tune hyperparameters for your GPU RAM):

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

4. Point the app or synthesis code at your **local** adapter/checkpoint directory if your integration expects a path under `model_output/Synthesis/finetuned/…`.

**If you do not fine-tune:** use the chatbot with **`ollama`** or **`vllm`** and a model you host yourself, or **`transformers`** with a smaller/public checkpoint — you still get DNA + Synapse from locally generated `model_output/` without a custom LoRA.

---

## Run the app

```bash
uv run streamlit run app/hk_etf_intelligence_app.py
```

### Tabs (short)

| Tab | Depends on |
|-----|----------------|
| **1W Winners/Losers** | OHLCV parquet files |
| **ETF Screener** | `ETP_Data_Export.xlsx` |
| **ETF Explorer** | OHLCV + top holdings |
| **AI Chatbot** | Synthesis backend + optional DNA/Synapse artifacts for evidence |

### Chatbot backends

| Backend | When to use |
|---------|-------------|
| **`transformers`** | In-process local model; first response pays model load cost. |
| **`ollama`** | Requires Ollama at `http://localhost:11434`. |
| **`vllm`** | Requires vLLM HTTP API (default expectation `http://localhost:8000`). |

---

## Pipeline commands

### Core ETF pipeline

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

### Financial DNA

```bash
uv run python src/model/dna/data_engine.py
uv run python src/model/dna/model_core.py
uv run python src/model/dna/advisory_logic.py
uv run python src/model/dna/visualize_clusters.py
```

All DNA steps:

```bash
uv run python src/model/dna/run.py
```

### Data fetchers (examples)

```bash
uv run python src/data_ingestion/provider/yfinance/etf_market_data_fetcher.py
uv run python src/data_ingestion/provider/yfinance/etf_top_holdings_data_fetcher.py
```

### Synapse

```bash
uv run python src/text_extraction/pdf_text_extractor.py
uv run python src/model/synapse/model.py --query "Fed pause supports duration-sensitive assets" --top-k 5
uv run python src/model/synapse/run.py
```

### Synthesis (no fine-tune)

```bash
uv run python src/model/synthesis/run.py
```

Quick check:

```bash
uv run python src/model/synthesis/run.py --ticker 2800 --query "What are key risks?"
```

---

## Project layout

```text
8017Group4.1/
├── app/
│   └── hk_etf_intelligence_app.py      # Streamlit UI
├── MODEL_DETAILS.md                     # Technical deep dive
├── README.md
├── pyproject.toml
├── data/                                # not in Git — you provide (see above)
│   └── etf/
│       ├── summary/
│       ├── ohlcv/
│       ├── holdings/top10/
│       └── documentation/<ticker>/{pdf,csv}/
├── model_output/                        # generated locally (DNA, Synapse, Synthesis)
│   ├── dna/
│   ├── synpse/
│   └── Synthesis/
│       ├── finetune_all/                # generated QnA JSONL
│       └── finetuned/                   # local LoRA / checkpoints (not from GitHub)
└── src/
    ├── etf_pipeline.py
    ├── data_ingestion/provider/
    ├── text_extraction/
    └── model/
        ├── dna/
        ├── synapse/
        └── synthesis/
```

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| Tabs empty or errors about missing files | Paths under `data/etf/…` and `model_output/…`; run pipelines to create DNA/Synapse outputs. |
| **`data/ETF` vs `data/etf`** | The app tries both; align your folder names if something still fails. |
| Chatbot slow on first message (`transformers`) | Normal: model loads once per process; later messages are faster. |
| **`ollama` / `vllm` errors** | Services must be running on the expected host/port. |
| Synapse / similarity odd results | Regenerate profiles and **`corpus_embeddings*.pt`** by running the Synapse pipeline locally. |
| Push to GitHub rejected for large files | Do not commit `data/`, huge checkpoints, or `corpus_embeddings*.pt`; use **Git LFS** only for allowed patterns in `.gitattributes` and stay within [LFS quotas](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). |

---

## Tech stack

- Python 3.10+
- pandas, numpy, scikit-learn  
- sentence-transformers, transformers, torch  
- Streamlit, Plotly  
- yfinance  

---

## Academic context

This project was developed in an HKU academic setting for ETF diversification research and investor-facing explainability.

---

## Disclaimer

This repository is for **academic and research purposes only**. It is **not** investment advice and not a recommendation to buy or sell any security.
