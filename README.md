# 🇭🇰 HK ETF Intelligence Platform

Evidence-driven ETF analytics and AI-style explanations for Hong Kong ETFs: clustering and advisory signals (**Financial DNA**), semantic links between disclosures and news (**Synapse**), and an advisor-style chat layer (**Financial Synthesis**). A **Streamlit** app ties screening, exploration, and chat into one UI.

---

## 📑 Table of contents

1. [📋 Overview](#overview)
2. [☁️ What’s on GitHub vs your machine](#whats-on-github-vs-your-machine)
3. [✅ Requirements](#requirements)
4. [🚀 Clone and install](#clone-and-install)
5. [📂 Data and Synapse caches](#data-and-synapse-caches)
6. [🎓 Fine-tune LoRA (optional)](#fine-tune-lora-optional)
7. [▶️ Run the app](#run-the-app)
8. [🛠️ Pipeline commands](#pipeline-commands)
9. [🗂️ Project layout](#project-layout)
10. [❓ Troubleshooting](#troubleshooting)
11. [🧰 Tech stack](#tech-stack)
12. [🎓 Academic context](#academic-context)
13. [⚠️ Disclaimer](#disclaimer)

---

## 📋 Overview

This project helps investors explore HK ETFs with three connected intelligence modules:

| Module | Role |
|--------|------|
| 🧬 **Financial DNA** | Feature engineering, PCA + K-Means clustering, advisory outputs (home-bias and “hidden twin” candidates). |
| 🔗 **Synapse** | ETF text profiles from prospectuses/key facts, **`corpus_embeddings*.pt`** caches, and similarity to news/events. |
| ✨ **Financial Synthesis** | Combines DNA + Synapse into readable answers with evidence; backends: **Transformers**, **Ollama**, or **vLLM**. |

📖 **Deeper technical write-up:** `MODEL_DETAILS.md` (repo root).

---

## ☁️ What’s on GitHub vs your machine

**Remote repo (GitHub)** ships **source code**, **`pyproject.toml`**, and the **Streamlit app** — not multi‑gigabyte datasets or Synapse tensor caches.

**On your laptop / lab machine** you normally keep (like our maintainers):

| ✅ You typically have locally | 💡 Role |
|-------------------------------|--------|
| 📂 **`data/`** | HKEX exports, OHLCV, holdings, documentation PDFs/CSVs — needed for pipelines and the app. |
| 🧠 **`corpus_embeddings*.pt`** + **`model_output/synpse/`** | Synapse caches and profiles — built after PDF/text + embedding runs; **not** the chat LLM weights. |
| 📊 **`model_output/dna/`**, **`model_output/synpse/news_events_run_*`** | DNA + Synapse outputs for evidence in synthesis. |

**🚫 The only big artifact we do *not* keep in Git (and you must train if you want it):**

| Missing from Git (by design) | Why |
|------------------------------|-----|
| 🧪 **`model_output/Synthesis/finetuned/…`** (LoRA adapters, **optimizer states**, **multi‑GB** trainer checkpoints) | Too large for GitHub; **custom Qwen/LoRA** is optional — see [Fine-tune LoRA (optional)](#fine-tune-lora-optional). |

> 💡 **TL;DR:** If you already have **`data/`** and **Synapse `corpus_embeddings*.pt`**, you’re in good shape for DNA + Synapse + the default chat (base Qwen or Ollama/vLLM). The **only** “extra” is **training your own LoRA** into `model_output/Synthesis/finetuned/…` if you want project-specific fine-tunes.

---

## ✅ Requirements

- **Python 3.10+** — see `pyproject.toml`
- **[uv](https://docs.astral.sh/uv/)** — `uv sync` from the repo root
- **Disk** — several GB for `data/` and `model_output/`; **much more** if you train a 7B LoRA (checkpoints, cache)
- **GPU** — 🎯 strongly recommended for local **Transformers** chat and **LoRA**; CPU can work for parts of DNA/Synapse

---

## 🚀 Clone and install

```bash
git clone git@github.com:Holicklis/8017Group4.1.git
cd 8017Group4.1

# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

---

## 📂 Data and Synapse caches

### Minimum paths for a full Streamlit experience

- `data/etf/summary/ETP_Data_Export.xlsx`
- `data/etf/ohlcv/<ticker>/ohlcv.parquet`
- `data/etf/holdings/top10/<ticker>/top_holdings.parquet`
- `model_output/dna/cluster_views/cluster_perspectives.parquet`
- `model_output/dna/advisory/home_bias_candidates.parquet`
- `model_output/dna/advisory/hidden_twin_candidates.parquet`
- `model_output/synpse/news_events_run_*/news_event_topk_matches.csv`

### Synapse embeds (`corpus_embeddings*.pt`)

These are **local cache files** produced when you run the Synapse side (profiles + sentence embeddings). They sit under paths such as `src/model/...` or `model_output/synpse/cache/` depending on your run — **keep them on disk**; they are **not** the same as a **fine-tuned Qwen** in `Synthesis/finetuned/`.

> 💡 If your tree uses **`data/ETF/`** instead of **`data/etf/`**, the app can fall back automatically.

---

## 🎓 Fine-tune LoRA (optional)

**Why this folder might be “missing”:** GitHub is not meant to store **multi‑GB** LoRA runs (`optimizer.pt`, many checkpoints). Everything under **`model_output/Synthesis/finetuned/`** is **trained locally** (or on a cloud GPU) when you want **custom** weights.

**Ways to train / obtain a LoRA run**

1. **🏠 Recommended — this repo’s script**  
   Generate ChatML data, then fine-tune Qwen with LoRA:

   ```bash
   # 1) QnA for one ticker (example)
   uv run python src/model/synthesis/generate_finetune_qa.py \
     --csv-dir "data/etf/documentation/02800/csv" \
     --output-dir "model_output/Synthesis/finetune" \
     --ticker 02800 \
     --max-pairs 240

   # 2) QnA for all tickers (adjust paths)
   uv run python src/model/synthesis/generate_finetune_qa.py \
     --all-csv \
     --documentation-root "data/etf/documentation" \
     --output-dir "model_output/Synthesis/finetune_all" \
     --max-pairs 40

   # 3) LoRA fine-tune (tune batch size / steps for your GPU RAM)
   uv run python src/model/synthesis/finetune_qwen.py \
     --dataset-path "model_output/Synthesis/finetune_all/all_tickers_finetune_qa_chatml.jsonl" \
     --output-dir "model_output/Synthesis/finetuned/qwen2.5_7b_lora_full" \
     --model-name "Qwen/Qwen2.5-7B-Instruct" \
     --epochs 1 \
     --batch-size 1 \
     --grad-accum 16 \
     --max-length 1024
   ```

   ⏱️ **Ballpark:** on an **Apple M2 Pro with 32 GB** RAM, a typical `finetune_qwen.py` run is on the order of **~10 hours** (varies with dataset size and flags — see the top of `src/model/synthesis/finetune_qwen.py`).

2. **☁️ Cloud GPU** — if your laptop is too small, run the same commands on **Google Colab**, **Kaggle Notebooks**, **Lambda / RunPod / vast.ai**, etc.: clone repo, upload `data/` + generated JSONL, run `finetune_qwen.py`, download **`model_output/Synthesis/finetuned/`** zip back to your machine.

3. **🔧 Other trainers** — any workflow that produces **Hugging Face–compatible LoRA** for `Qwen2.5-7B-Instruct` can work in principle; you’d still point outputs under `model_output/Synthesis/finetuned/…` for consistency. (The Streamlit **`transformers`** path loads **base** models by default — wiring **Peft** adapters into chat would need a small code change if you want live LoRA inference.)

**If you skip LoRA:** use the chatbot with **`ollama`** or **`vllm`**, or **`transformers`** + **`Qwen/Qwen2.5-7B-Instruct`** — you still get DNA + Synapse evidence from your local **`model_output/`**.

---

## ▶️ Run the app

```bash
uv run streamlit run app/hk_etf_intelligence_app.py
```

### 🗂️ Tabs

| Tab | Depends on |
|-----|------------|
| 📈 **1W Winners/Losers** | OHLCV parquet files |
| 🔍 **ETF Screener** | `ETP_Data_Export.xlsx` |
| 🗺️ **ETF Explorer** | OHLCV + top holdings |
| 🤖 **AI Chatbot** | Synthesis backend + DNA/Synapse artifacts for evidence |

### 🔌 Chatbot backends

| Backend | When to use |
|---------|-------------|
| **`transformers`** | In-process local model; first reply pays load time. |
| **`ollama`** | Local Ollama at `http://localhost:11434` |
| **`vllm`** | vLLM HTTP API (e.g. `http://localhost:8000`) |

---

## 🛠️ Pipeline commands

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

All DNA steps:

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
uv run python src/text_extraction/pdf_text_extractor.py
uv run python src/model/synapse/model.py --query "Fed pause supports duration-sensitive assets" --top-k 5
uv run python src/model/synapse/run.py
```

### ✨ Synthesis (CLI)

```bash
uv run python src/model/synthesis/run.py
```

Quick test:

```bash
uv run python src/model/synthesis/run.py --ticker 2800 --query "What are key risks?"
```

---

## 🗂️ Project layout

```text
8017Group4.1/
├── app/
│   └── hk_etf_intelligence_app.py     # 🖥️ Streamlit UI
├── MODEL_DETAILS.md
├── README.md
├── pyproject.toml
├── data/                               # 📂 local ETF data (not on GitHub)
│   └── etf/
│       ├── summary/
│       ├── ohlcv/
│       ├── holdings/top10/
│       └── documentation/<ticker>/{pdf,csv}/
├── model_output/                       # 🔧 generated locally
│   ├── dna/
│   ├── synpse/                         # + corpus_embeddings caches
│   └── Synthesis/
│       ├── finetune_all/               # QnA / ChatML from scripts
│       └── finetuned/                  # 🧪 optional LoRA output (train locally)
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

## ❓ Troubleshooting

| 😕 Issue | ✅ What to check |
|----------|------------------|
| Tabs empty / missing file errors | `data/etf/…` and `model_output/…` paths; run pipelines. |
| **`data/ETF` vs `data/etf`** | App tries both; align names if something still fails. |
| Chatbot slow on first message (`transformers`) | Normal — model loads once; later replies are faster. |
| **`ollama` / `vllm` errors** | Services running on expected host/port. |
| Synapse looks off | Re-run Synapse pipeline; refresh **`corpus_embeddings*.pt`** / profiles. |
| Git push rejected — file too large | Don’t commit **`data/`**, **`corpus_embeddings*.pt`**, or **`model_output/Synthesis/finetuned/`** — keep them local or share via team storage / releases. |

---

## 🧰 Tech stack

- Python 3.10+
- pandas, numpy, scikit-learn
- sentence-transformers, transformers, torch
- Streamlit, Plotly
- yfinance

---

## 🎓 Academic context

This project was developed in an HKU academic setting for ETF diversification research and investor-facing explainability.

---

## ⚠️ Disclaimer

This repository is for **academic and research purposes only**. It is **not** investment advice and not a recommendation to buy or sell any security.
