#!/usr/bin/env python3
"""
Populate data/ per README.md and notebooks/01_preprocessing.ipynb.

  • Complaints, Phrasebank, Financial Q&A — public mirrors (no Kaggle).
  • Alpha Insights, Finance survey, S&P500 prices, Financial News, FinSen — Kaggle API.

Kaggle auth (pick one):
  - New API tokens (KGAT_…): export KAGGLE_API_TOKEN=…
  - Legacy: ./kaggle.json or ~/.kaggle/kaggle.json with username + key, or KAGGLE_USERNAME + KAGGLE_KEY

Run from repo root:
  python3 scripts/download_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

COMPLAINTS_COL_MAP = {
    "complaint_id": "Complaint ID",
    "product": "Product",
    "issue": "Issue",
    "sub_issue": "Sub-issue",
}


def bootstrap_kaggle_json() -> None:
    """Copy project-root kaggle.json to ~/.kaggle if home is missing it."""
    kdir = Path.home() / ".kaggle"
    kfile = kdir / "kaggle.json"
    if kfile.exists():
        return
    proj = BASE / "kaggle.json"
    if not proj.exists():
        return
    kdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(proj, kfile)
    os.chmod(kfile, 0o600)


def has_kaggle_credentials() -> bool:
    if (os.environ.get("KAGGLE_API_TOKEN") or "").strip():
        return True
    bootstrap_kaggle_json()
    kfile = Path.home() / ".kaggle" / "kaggle.json"
    if kfile.exists():
        return True
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        kdir = Path.home() / ".kaggle"
        kdir.mkdir(parents=True, exist_ok=True)
        kfile.write_text(json.dumps({"username": user, "key": key}))
        os.chmod(kfile, 0o600)
        return True
    return False


def require_kaggle_credentials() -> None:
    if not has_kaggle_credentials():
        print(
            "Missing Kaggle credentials for some datasets.\n"
            "  export KAGGLE_API_TOKEN=KGAT_…   (recommended)\n"
            "  or kaggle.json in project root / ~/.kaggle/, or KAGGLE_USERNAME + KAGGLE_KEY.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def ensure_pip_deps() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "kaggle", "huggingface_hub", "pandas", "pyarrow"]
    )


def kaggle_executable() -> str:
    """Kaggle 2.x registers a `kaggle` script; `python -m kaggle` is not supported."""
    exe = shutil.which("kaggle")
    if exe:
        return exe
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    for base in (
        Path.home() / "Library" / "Python" / ver / "bin",
        Path(sys.executable).resolve().parent,
    ):
        cand = base / "kaggle"
        if cand.is_file():
            return str(cand)
    raise RuntimeError(
        "Kaggle CLI not found after pip install. Add Python's scripts directory to PATH, "
        'or run: python3 -m pip install kaggle && export PATH="$HOME/Library/Python/'
        f'{ver}/bin:$PATH"'
    )


def run_kaggle_download(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tpath = Path(tmp)
        cmd = [
            kaggle_executable(),
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(tpath),
            "--unzip",
            "--force",
        ]
        print("  ", " ".join(cmd[1:]))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr, sep="", file=sys.stderr)
            raise RuntimeError(f"Kaggle download failed: {slug}")
        seen: set[str] = set()
        for csv_path in sorted(tpath.rglob("*.csv")):
            name = csv_path.name
            if name in seen:
                stem, suf = csv_path.stem, csv_path.suffix
                for i in range(1, 10_000):
                    alt = f"{stem}_{i}{suf}"
                    if alt not in seen:
                        name = alt
                        break
            seen.add(name)
            shutil.move(str(csv_path), str(dest / name))


def download_complaints_hf(num_shards: int = 2) -> None:
    """HF mirror of CFPB-style complaints; column names aligned to Kaggle CSV."""
    from huggingface_hub import hf_hub_download

    dest = DATA / "complaints"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "customer_complaints.csv"
    if out.exists() and out.stat().st_size > 1_000_000:
        print("Complaints: already present, skipping.")
        return

    frames = []
    for i in range(num_shards):
        path = hf_hub_download(
            "davidheineman/consumer-finance-complaints-large",
            f"data/train-{i:05d}-of-00011.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        df = df.rename(columns={k: v for k, v in COMPLAINTS_COL_MAP.items() if k in df.columns})
        # Keep columns notebook expects (ignore extra columns)
        need = ["Complaint ID", "Product", "Issue", "Sub-issue"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"Complaints parquet missing columns {missing}")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(out, index=False)
    print(f"Complaints: wrote {len(merged):,} rows → {out.relative_to(BASE)}")


def download_phrasebank_hf() -> None:
    from huggingface_hub import hf_hub_download

    dest = DATA / "phrasebank"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "all-data.csv"
    if out.exists():
        print("Phrasebank: already present, skipping.")
        return
    path = hf_hub_download(
        "warwickai/financial_phrasebank_mirror",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    mp = {0: "negative", 1: "neutral", 2: "positive"}
    export = pd.DataFrame(
        {"sentiment": df["label"].map(mp), "sentence": df["sentence"]}
    )
    export.to_csv(out, header=False, index=False, encoding="latin-1")
    print(f"Phrasebank: {len(export):,} rows → {out.relative_to(BASE)}")


def download_financial_qa_hf() -> None:
    from huggingface_hub import hf_hub_download

    dest = DATA / "financial-qa"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "Financial-QA-10k.csv"
    if out.exists():
        print("Financial Q&A: already present, skipping.")
        return
    path = hf_hub_download(
        "virattt/financial-qa-10K",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    qa = pd.read_parquet(path)
    cols = [c for c in ("question", "answer", "ticker", "context", "filing") if c in qa.columns]
    qa[cols].to_csv(out, index=False)
    print(f"Financial Q&A: {len(qa):,} rows → {out.relative_to(BASE)}")


def organize_kaggle_outputs() -> None:
    """Normalize occasional filename variants after Kaggle unzip."""

    qdir = DATA / "financial-qa"
    tq = qdir / "Financial-QA-10k.csv"
    if not tq.exists():
        for p in qdir.glob("*.csv"):
            shutil.copy2(p, tq)
            break

    pdir = DATA / "phrasebank"
    tp = pdir / "all-data.csv"
    if not tp.exists():
        for name in ("all-data.csv", "AllAgree.csv"):
            hit = pdir / name
            if hit.exists():
                shutil.copy2(hit, tp)
                break

    fdir = DATA / "finance-data"
    fdir.mkdir(parents=True, exist_ok=True)
    tf = fdir / "Finance_data.csv"
    if not tf.exists():
        for p in fdir.glob("*.csv"):
            shutil.copy2(p, tf)
            break

    ndir = DATA / "financial-news"
    tn = ndir / "financial_news_events.csv"
    if not tn.exists():
        for p in ndir.glob("*.csv"):
            shutil.copy2(p, tn)
            break

    fsdir = DATA / "finsen"
    tfs = fsdir / "FinSen_US_Categorized.csv"
    if not tfs.exists():
        for p in fsdir.glob("*.csv"):
            if "finsen" in p.name.lower() or "categor" in p.name.lower():
                shutil.copy2(p, tfs)
                break
        if not tfs.exists() and list(fsdir.glob("*.csv")):
            shutil.copy2(max(fsdir.glob("*.csv"), key=lambda x: x.stat().st_size), tfs)

    adir = DATA / "alpha-insights"
    for req in ("ETFs.csv", "MutualFunds.csv"):
        if (adir / req).exists():
            continue
        needle = "etf" if req.startswith("ETF") else "mutual"
        for f in sorted(adir.glob("*.csv")):
            if needle in f.stem.lower():
                shutil.copy2(f, adir / req)
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project datasets into data/.")
    parser.add_argument(
        "--complaint-shards",
        type=int,
        default=2,
        help="HF consumer-finance parquet shards (each ~650k rows). Default: 2",
    )
    parser.add_argument(
        "--kaggle-only",
        action="store_true",
        help="Only use Kaggle for all eight (requires credentials).",
    )
    args = parser.parse_args()

    ensure_pip_deps()

    for s in (
        "alpha-insights",
        "complaints",
        "phrasebank",
        "finance-data",
        "financial-qa",
        "sp500-etf-crypto",
        "financial-news",
        "finsen",
        "processed",
    ):
        (DATA / s).mkdir(parents=True, exist_ok=True)

    if args.kaggle_only:
        require_kaggle_credentials()
        jobs = [
            ("willianoliveiragibin/alpha-insights-us-funds", DATA / "alpha-insights"),
            ("taeefnajib/bank-customer-complaints", DATA / "complaints"),
            ("ankurzing/sentiment-analysis-for-financial-news", DATA / "phrasebank"),
            ("nitindatta/finance-data", DATA / "finance-data"),
            ("yousefsaeedian/financial-q-and-a-10k", DATA / "financial-qa"),
            ("benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated", DATA / "sp500-etf-crypto"),
            ("pratyushpuri/financial-news-market-events-dataset-2025", DATA / "financial-news"),
            ("eaglewhl/finsen-financial-sentiment-dataset", DATA / "finsen"),
        ]
        for slug, dest in jobs:
            print(f"Kaggle: {slug} → {dest.relative_to(BASE)}")
            run_kaggle_download(slug, dest)
        organize_kaggle_outputs()
        print("Done (Kaggle-only). Run notebooks/01_preprocessing.ipynb next.")
        return

    # Public mirrors first (no Kaggle)
    print("Downloading complaints (Hugging Face, CFPB-style) …")
    download_complaints_hf(num_shards=args.complaint_shards)
    print("Downloading phrasebank (Hugging Face) …")
    download_phrasebank_hf()
    print("Downloading financial Q&A (Hugging Face) …")
    download_financial_qa_hf()

    kaggle_jobs = [
        ("willianoliveiragibin/alpha-insights-us-funds", DATA / "alpha-insights"),
        ("nitindatta/finance-data", DATA / "finance-data"),
        ("benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated", DATA / "sp500-etf-crypto"),
        ("pratyushpuri/financial-news-market-events-dataset-2025", DATA / "financial-news"),
        ("eaglewhl/finsen-financial-sentiment-dataset", DATA / "finsen"),
    ]

    print("Remaining datasets: alpha-insights, finance-data, sp500 prices, financial-news, finsen (Kaggle).")
    if not has_kaggle_credentials():
        print(
            "\nNo Kaggle API token found — stopped after public mirrors.\n"
            "Add kaggle.json (see script docstring), then run this script again to fetch the rest.\n"
            "Or use: python3 scripts/download_data.py --kaggle-only\n"
        )
        return

    for slug, dest in kaggle_jobs:
        print(f"Kaggle: {slug} → {dest.relative_to(BASE)}")
        run_kaggle_download(slug, dest)

    organize_kaggle_outputs()
    print("Done. Run notebooks/01_preprocessing.ipynb to build data/processed/.")


if __name__ == "__main__":
    main()
