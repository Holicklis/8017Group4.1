# HK ETF Data Pipeline

This repository includes an ETF data engineering workflow for:

1. Exporting ETF metadata from HKEX.
2. Downloading ETF-related PDF documents (prospectus and related filings).
3. Converting downloaded PDFs into cleaned sentence-level CSV files.

The current refactoring keeps existing script entrypoints runnable while improving naming consistency, maintainability, and path robustness.

## Naming Convention

The project now uses a consistent naming standard:

- folders: `lowercase_snake_case`
- python files: `lowercase_snake_case.py`
- generated ETF artifacts: `data/etf/...`

Legacy uppercase paths are still supported in code for backward compatibility, but all new work should use lowercase paths.

## What This Pipeline Produces

- **Metadata export**: `data/etf/summary/ETP_Data_Export.xlsx`
- **Downloaded PDF documents**: `data/etf/documentation/<TICKER>/pdf/*.pdf`
- **Parsed text CSV files**: `data/etf/documentation/<TICKER>/csv/*.csv`

## Source Modules

- `src/hkex_etf/etf_metadata_export.py`  
  Downloads HKEX ETP summary export file.

- `src/hkex_etf/etf_document_scraper.py`  
  Scrapes HKEX quote/news pages and downloads ETF PDFs.

- `src/text_extraction/pdf_text_extractor.py`  
  Extracts and cleans PDF text, then saves sentence-level CSV.

- `src/etf_pipeline.py`  
  New orchestrator entrypoint to run all stages in sequence.

## Project Layout

```text
8017Group4.1/
├── data/
│   └── etf/
│       ├── summary/
│       │   └── ETP_Data_Export.xlsx
│       └── documentation/
│           └── 02801/
│               ├── pdf/
│               └── csv/
├── src/
│   ├── etf_pipeline.py
│   ├── hkex_etf/
│   │   ├── etf_metadata_export.py
│   │   └── etf_document_scraper.py
│   ├── text_extraction/
│   │   └── pdf_text_extractor.py
│   ├── market_data/
│   └── model/
└── README.md
```

## Dependency Management (uv)

This project now uses `uv` and `pyproject.toml` (not `requirements.txt`).

### 1) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2) Create and sync environment

```bash
uv sync
```

### 3) Run commands with uv

```bash
uv run python src/etf_pipeline.py
```

### Common uv commands

```bash
uv add <package>        # add runtime dependency
uv add --dev <package>  # add dev dependency
uv lock                 # refresh lockfile
uv sync                 # install from lock
uv run <command>        # run command in managed env
```

## Run Options

### Option A: Run complete pipeline (recommended)

```bash
uv run python src/etf_pipeline.py
```

Useful flags:

```bash
uv run python src/etf_pipeline.py --skip-export
uv run python src/etf_pipeline.py --skip-documents
uv run python src/etf_pipeline.py --skip-text-extract
uv run python src/etf_pipeline.py --summary-file "data/etf/summary/ETP_Data_Export.xlsx"
uv run python src/etf_pipeline.py --no-headless
```

### Option B: Run each stage directly

Metadata export:

```bash
uv run python src/hkex_etf/etf_metadata_export.py
```

PDF scraping (expects `instruments.csv` in working directory when run directly):

```bash
uv run python src/hkex_etf/etf_document_scraper.py
```

PDF text extraction:

```bash
uv run python src/text_extraction/pdf_text_extractor.py
```

## Refactoring Notes

- **Single canonical script layout**: duplicate wrapper modules were removed to avoid parallel implementations.
- **Improved path dependency**: script paths now resolve from file location instead of current working directory whenever possible.
- **Improved maintainability**:
  - Added structured logging support.
  - Added/expanded type hints and function docstrings.
  - Improved variable naming clarity (`df_*` naming for DataFrames).
- **Pipeline orchestration**: added `src/etf_pipeline.py` to make the data flow executable end-to-end in one command.
- **Naming consistency**: standardized primary directories/files to lowercase snake_case.

## Logging

- Logs are written under `logs/` instead of project root.
- Document scraping logs: `logs/etp_prospectus.log`.
- PDF extraction logs: `logs/pdf_text_extractor.log`.

## Operational Notes

- Selenium requires a compatible Chrome/Chromium and driver setup.
- HKEX page structure may evolve. If selectors fail, update the CSS/XPath selectors in scraper modules.
- Downloaded file names for PDFs include ticker + timestamp + index + short title to reduce collisions.

## Disclaimer

This project is for academic/research use and should not be considered financial advice.

## DNA 🧬 Model (Financial DNA)

The DNA 🧬 model is a 3-module ETF intelligence workflow that converts market + fund data into explainable clustering insights.

### Core Idea

- **Module 1 (`src/model/dna/data_engine.py`)** builds Financial DNA features from metadata and OHLCV time series.
- **Module 2 (`src/model/dna/model_core.py`)** creates multiple PCA + KMeans cluster perspectives (risk/return, income, liquidity, macro).
- **Module 3 (`src/model/dna/advisory_logic.py`)** generates actionable signals such as home-bias candidates and hidden twins.

This structure separates:
- data cleaning/feature creation,
- statistical grouping,
- decision-support logic.

### DNA 🧬 Inputs and Outputs

Inputs:
- Metadata: `data/etf/summary/ETP_Data_Export.xlsx`
- Instrument universe: `data/etf/instruments/all_hk_etf.csv`
- OHLCV parquet files: `data/etf/ohlcv/...`

Main outputs:
- `data/etf/processed/financial_dna.parquet`
- `data/etf/processed/cluster_views/cluster_perspectives.parquet`
- `data/etf/processed/advisory/home_bias_candidates.parquet`
- `data/etf/processed/advisory/hidden_twin_candidates.parquet`

### How to Run the DNA 🧬 Model

Step 1: build Financial DNA features

```bash
uv run python src/model/dna/data_engine.py
```

Step 2: run multi-perspective PCA + clustering

```bash
uv run python src/model/dna/model_core.py
```

Step 3: generate advisory signals

```bash
uv run python src/model/dna/advisory_logic.py
```

Recommended strict filtering (fewer pairs):

```bash
uv run python src/model/dna/advisory_logic.py \
  --min-label-mismatches 3 \
  --max-pc-distance 1.2 \
  --top-k-per-etf 3 \
  --home-bias-max-pc-distance 1.2 \
  --home-bias-top-k-per-etf 3
```

### Cluster Visualization Script

Use the visualization script to inspect cluster geometry and cluster size balance:

```bash
uv run python src/model/dna/visualize_clusters.py
```

Optional flags:

```bash
uv run python src/model/dna/visualize_clusters.py --annotate-points
uv run python src/model/dna/visualize_clusters.py --input-path "data/etf/processed/cluster_views/cluster_perspectives.parquet"
uv run python src/model/dna/visualize_clusters.py --output-dir "data/etf/processed/cluster_views/plots"
```

Visualization outputs:
- Scatter plots by perspective: `*_pc_scatter.png`
- Cluster size plots by perspective: `*_cluster_sizes.png`
- Cluster size summary table: `cluster_size_summary.csv`