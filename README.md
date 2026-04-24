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