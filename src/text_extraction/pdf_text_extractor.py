import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pypdf
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from tqdm import tqdm

# Configure Logging
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_DIR / "pdf_text_extractor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ETFPDFProcessor")


class ETFPDFProcessor:
    """
    A professional pipeline to extract, clean, and structure ETF prospectus data
    from PDF files into CSV format for LLM fine-tuning and vector embeddings.
    """

    PROFILE_VERSION = "etf_profile_v2_kfs_focused"
    SECTION_CHAR_BUDGET = 1200
    GLOBAL_CHAR_BUDGET = 2600
    SECTION_SENTENCE_BUDGET = 16
    GLOBAL_SENTENCE_BUDGET = 32
    RISK_SUMMARY_SENTENCE_BUDGET = 6
    RISK_SUMMARY_CHAR_BUDGET = 650
    PROFILE_OUTPUT_RELATIVE_PATH = Path("model_output/synpse/etf_profiles.csv")
    DOC_TYPE_WEIGHT = {"key_facts": 5, "prospectus": 2, "other": 0}
    RISK_SECTION_HINTS = (
        "risk factors",
        "key risks",
        "risk profile",
        "investment risks",
        "risk",
    )
    RISK_NOISE_PHRASES = (
        "quick facts",
        "stock code",
        "trading lot size",
        "manager:",
        "trustee",
        "registrar",
        "counterparty",
        "comply with the following provisions",
        "the following provisions",
        "subject to paragraphs",
        "otc",
        "over-the-counter",
        "appendix",
        "schedule",
        "table",
    )

    RISK_KEYWORDS = {
        "interest_rate": [
            "interest rate",
            "rate risk",
            "monetary policy",
            "hkma",
            "federal reserve",
            "fed funds",
            "yield",
            "duration risk",
        ],
        "commodity": [
            "commodity",
            "oil",
            "crude",
            "gold",
            "silver",
            "metals",
            "natural gas",
            "energy price",
            "commodity volatility",
        ],
        "equity_market_hk": [
            "hong kong equity",
            "hang seng",
            "main board",
            "equity market",
            "market selloff",
            "valuation risk",
        ],
        "equity_fundamental": [
            "earnings",
            "earnings growth",
            "earnings revision",
            "revenue",
            "sales growth",
            "profit margin",
            "guidance cut",
            "valuation multiple",
            "price to earnings",
            "p/e",
        ],
        "region_risk": [
            "apac",
            "amer",
            "emea",
            "asean",
            "north america",
            "latin america",
            "europe",
            "middle east",
            "africa",
            "japan",
            "korea",
            "taiwan",
            "india",
            "emerging market",
            "frontier market",
            "regional risk",
            "cross border risk",
        ],
        "tech_sector": [
            "technology",
            "tech sector",
            "ai",
            "artificial intelligence",
            "semiconductor",
            "chip",
            "hardware",
            "software",
            "cloud computing",
            "cybersecurity",
            "data center",
        ],
        "us_labor_macro": [
            "nonfarm payroll",
            "unemployment",
            "labor market",
            "wage growth",
            "jobs report",
        ],
        "us_specific_risk": [
            "us recession",
            "debt ceiling",
            "treasury yield",
            "fomc",
            "federal reserve",
            "fiscal cliff",
            "government shutdown",
            "us policy risk",
        ],
        "china_specific_risk": [
            "china",
            "prc",
            "mainland",
            "pbo c",
            "pboc",
            "property sector",
            "local government debt",
            "geopolitical tension",
            "regulatory crackdown",
            "china policy",
            "capital control",
        ],
        "money_market": [
            "money market",
            "treasury bill",
            "t-bill",
            "commercial paper",
            "certificate of deposit",
            "repo",
            "liquidity buffer",
            "cash equivalent",
            "short duration",
            "short term rate",
        ],
        "style_factor": [
            "value",
            "quality",
            "high dividend",
            "growth",
            "large cap",
            "mid cap",
            "small cap",
            "passive",
            "active",
            "esg",
            "sustainable",
            "leverage",
            "factor",
            "low volatility",
            "momentum",
            "size factor",
        ],
        "fx": ["foreign exchange", "currency risk", "fx risk", "exchange rate"],
        "credit_liquidity": [
            "credit risk",
            "liquidity risk",
            "counterparty",
            "default risk",
        ],
        "tracking_error": ["tracking error", "tracking difference", "index tracking"],
        "concentration": [
            "concentration risk",
            "single issuer",
            "sector concentration",
        ],
        "derivatives": ["derivatives", "swap", "futures", "options", "synthetic"],
    }
    COMPONENT_KEYWORDS = {
        "benchmark_index": [
            "benchmark",
            "index",
            "tracked",
            "tracking",
            "index provider",
        ],
        "asset_class": [
            "asset class",
            "equity",
            "bond",
            "money market",
            "commodity",
            "fixed income",
        ],
        "geographic_focus": [
            "geographic focus",
            "hong kong",
            "china",
            "asia",
            "us",
            "global",
            "emerging market",
        ],
        "replication_method": [
            "full replication",
            "representative sampling",
            "synthetic",
            "physical",
        ],
        "strategy": [
            "investment objective",
            "investment strategy",
            "objective",
            "portfolio",
            "constituent",
        ],
        "distribution": ["distribution policy", "dividend policy"],
    }

    def __init__(self, data_root: Optional[Union[str, Path]] = None):
        """
        Initializes the processor with path management.

        Args:
            data_root: Path to the 'data' directory. If None, it resolves
                       relative to this script's location.
        """
        if data_root:
            self.data_dir = Path(data_root)
        else:
            # Script is in src/text_extraction/, data is in ../../data/
            self.data_dir = Path(__file__).resolve().parents[2] / "data"

        lower_path = self.data_dir / "etf" / "documentation"
        legacy_paths = [self.data_dir / "ETF" / "documentation"]
        self.etf_doc_path = lower_path
        if not self.etf_doc_path.exists():
            for path in legacy_paths:
                if path.exists():
                    self.etf_doc_path = path
                    break

        if not self.data_dir.exists():
            logger.error(f"Data directory not found at: {self.data_dir}")
        else:
            logger.info(f"Initialized ETFPDFProcessor. Data Root: {self.data_dir}")

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _looks_like_noise(self, text: str) -> bool:
        text_lower = text.lower()
        if len(text_lower) < 30:
            return True
        if text_lower.count("http") >= 1 or "www." in text_lower:
            return True
        if "table of contents" in text_lower:
            return True
        if text_lower.count("restricted") >= 2:
            return True
        if re.search(r"\b(page|appendix|chapter)\s+\d+\b", text_lower):
            return True
        if sum(1 for phrase in self.RISK_NOISE_PHRASES if phrase in text_lower) >= 2:
            return True
        if re.search(r"\([a-z]\)\s", text_lower) and text_lower.count(";") >= 2:
            return True
        # Avoid rows that are mostly numbers/codes from PDF artifacts
        alnum = re.sub(r"\s+", "", text_lower)
        if alnum and sum(char.isdigit() for char in alnum) / len(alnum) > 0.35:
            return True
        return False

    def _clean_text_segment(self, text: str) -> str:
        """
        Standard text data cleaning: removes noise, artifacts, and formatting issues.
        """
        if not text:
            return ""

        # 1. Remove dot leaders (....) common in Tables of Contents
        text = re.sub(r"\.{2,}", " ", text)

        # 2. Remove Page Indicators (e.g., "- i -", "- 1 -", "- 22 -")
        text = re.sub(r"-\s*[ivx0-9]+\s*-", "", text, flags=re.IGNORECASE)

        # 3. Standardize whitespace (remove newlines/tabs)
        text = text.replace("\n", " ").replace("\t", " ")

        # 4. Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _merge_fragments(self, sentences: List[str]) -> List[str]:
        """
        Heuristic to rejoin sentences split by PDF line/page breaks.
        """
        merged = []
        current_buffer = ""

        terminal_punctuation = {".", "!", "?", "”", '"'}

        for segment in sentences:
            if not segment:
                continue

            if not current_buffer:
                current_buffer = segment
            else:
                # If current buffer doesn't end in punctuation, the next segment is a continuation
                if current_buffer[-1] not in terminal_punctuation:
                    current_buffer += " " + segment
                else:
                    merged.append(current_buffer)
                    current_buffer = segment

        if current_buffer:
            merged.append(current_buffer)

        return merged

    def _doc_type(self, file_path: Path) -> str:
        name = file_path.name.lower()
        if "key_facts" in name or "key facts" in name or "kfs" in name:
            return "key_facts"
        if "product_key_facts" in name or "product key facts" in name:
            return "key_facts"
        if "prospectus" in name:
            return "prospectus"
        return "other"

    def _contains_risk_focus(self, text: str) -> bool:
        text_lower = text.lower()
        if any(hint in text_lower for hint in self.RISK_SECTION_HINTS):
            return True
        # Common investor-facing risk language.
        investor_risk_patterns = (
            "subject to",
            "may be adversely",
            "volatility",
            "tracking error",
            "concentration risk",
            "liquidity risk",
            "currency risk",
            "credit risk",
            "interest rate risk",
            "counterparty risk",
        )
        return any(pattern in text_lower for pattern in investor_risk_patterns)

    def _normalize_summary_sentence(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip(" .;,:")
        if not text:
            return ""
        if len(text) > 220:
            text = text[:220].rsplit(" ", 1)[0]
        return text + "."

    def _extract_tags(self, text: str, keyword_map: Dict[str, List[str]]) -> Set[str]:
        text_lower = text.lower()
        tags: Set[str] = set()
        for tag, keywords in keyword_map.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.add(tag)
        return tags

    def _score_sentence(self, text: str, keyword_map: Dict[str, List[str]]) -> int:
        text_lower = text.lower()
        score = 0
        for keywords in keyword_map.values():
            score += sum(1 for keyword in keywords if keyword in text_lower)
        return score

    def _extract_structured_field(self, candidates: List[str], patterns: List[str]) -> str:
        for sentence in candidates:
            for pattern in patterns:
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match:
                    value = match.group(1).strip(" .;,:")
                    if 3 <= len(value) <= 140:
                        return value
        return ""

    def _select_ranked_sentences(
        self,
        candidates: List[Dict[str, Union[str, int]]],
        sentence_limit: int,
        char_budget: int,
        seen_keys: Set[str],
    ) -> List[str]:
        selected: List[str] = []
        used_chars = 0
        for row in sorted(candidates, key=lambda item: int(item["score"]), reverse=True):
            sentence = str(row["text"]).strip()
            normalized = self._normalize_text(sentence)
            if not sentence or normalized in seen_keys:
                continue
            if len(normalized.split()) < 8:
                continue
            # Near-duplicate control: compare first 18 normalized tokens.
            prefix_key = " ".join(normalized.split()[:18])
            if prefix_key in seen_keys:
                continue
            if used_chars + len(sentence) > char_budget:
                continue
            selected.append(sentence)
            used_chars += len(sentence)
            seen_keys.add(normalized)
            seen_keys.add(prefix_key)
            if len(selected) >= sentence_limit:
                break
        return selected

    def _build_ticker_profile(self, ticker_dir: Path) -> Optional[Dict[str, str]]:
        csv_files = sorted((ticker_dir / "csv").glob("*.csv"))
        if not csv_files:
            return None

        ranked_risk: List[Dict[str, Union[str, int]]] = []
        ranked_component: List[Dict[str, Union[str, int]]] = []
        all_candidate_sentences: List[str] = []
        risk_tags: Set[str] = set()
        component_tags: Set[str] = set()

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            if "text" not in df.columns:
                continue

            document_type = self._doc_type(csv_file)
            doc_weight = self.DOC_TYPE_WEIGHT.get(document_type, 0)
            if doc_weight <= 0:
                continue
            for raw_sentence in df["text"].dropna().astype(str).tolist():
                sentence = self._clean_text_segment(raw_sentence)
                if self._looks_like_noise(sentence):
                    continue
                if len(sentence) > 420:
                    continue

                risk_score = self._score_sentence(sentence, self.RISK_KEYWORDS) * doc_weight
                component_score = self._score_sentence(sentence, self.COMPONENT_KEYWORDS) * doc_weight
                if risk_score == 0 and component_score == 0:
                    continue

                all_candidate_sentences.append(sentence)
                risk_tags |= self._extract_tags(sentence, self.RISK_KEYWORDS)
                component_tags |= self._extract_tags(sentence, self.COMPONENT_KEYWORDS)
                if risk_score > 0 and self._contains_risk_focus(sentence):
                    ranked_risk.append({"text": sentence, "score": risk_score})
                if component_score > 0:
                    ranked_component.append({"text": sentence, "score": component_score})

        if not all_candidate_sentences:
            return None

        seen_keys: Set[str] = set()
        selected_components = self._select_ranked_sentences(
            candidates=ranked_component,
            sentence_limit=self.SECTION_SENTENCE_BUDGET,
            char_budget=self.SECTION_CHAR_BUDGET,
            seen_keys=seen_keys,
        )
        selected_risks = self._select_ranked_sentences(
            candidates=ranked_risk,
            sentence_limit=self.RISK_SUMMARY_SENTENCE_BUDGET,
            char_budget=self.RISK_SUMMARY_CHAR_BUDGET,
            seen_keys=seen_keys,
        )

        ordered_sentences = selected_components + selected_risks
        final_sentences: List[str] = []
        global_chars = 0
        for sentence in ordered_sentences:
            if len(final_sentences) >= self.GLOBAL_SENTENCE_BUDGET:
                break
            if global_chars + len(sentence) > self.GLOBAL_CHAR_BUDGET:
                continue
            final_sentences.append(sentence)
            global_chars += len(sentence)

        if not final_sentences:
            return None

        benchmark = self._extract_structured_field(
            all_candidate_sentences,
            [
                r"(?:benchmark|index)\s*[:\-]\s*([^.;]{5,120})",
                r"closely correspond to the performance of the\s+([^.;]{5,120})",
            ],
        )
        asset_class = self._extract_structured_field(
            all_candidate_sentences,
            [r"asset class\s*[:\-]\s*([^.;]{3,80})"],
        )
        geographic_focus = self._extract_structured_field(
            all_candidate_sentences,
            [r"geographic focus\s*[:\-]\s*([^.;]{3,80})"],
        )

        key_components = " ".join(selected_components).strip()
        normalized_risk_sentences = [self._normalize_summary_sentence(s) for s in selected_risks]
        normalized_risk_sentences = [s for s in normalized_risk_sentences if s]
        key_risks = " ".join(normalized_risk_sentences).strip()
        profile_text_parts = [
            f"Ticker: {ticker_dir.name}.",
            f"Benchmark/Index: {benchmark or 'not clearly stated'}.",
            f"Asset class: {asset_class or 'not clearly stated'}.",
            f"Geographic focus: {geographic_focus or 'not clearly stated'}.",
            f"Key components: {key_components}",
            f"Key risks: {key_risks}",
            f"Risk tags: {', '.join(sorted(risk_tags)) or 'none'}.",
            f"Component tags: {', '.join(sorted(component_tags)) or 'none'}.",
        ]
        profile_text = " ".join(part for part in profile_text_parts if part).strip()
        if len(profile_text) > self.GLOBAL_CHAR_BUDGET:
            profile_text = profile_text[: self.GLOBAL_CHAR_BUDGET].rsplit(" ", 1)[0]

        return {
            "ticker": ticker_dir.name,
            "benchmark_or_index": benchmark,
            "asset_class": asset_class,
            "geographic_focus": geographic_focus,
            "key_risks": key_risks[: self.SECTION_CHAR_BUDGET],
            "key_components": key_components[: self.SECTION_CHAR_BUDGET],
            "risk_tags": ",".join(sorted(risk_tags)),
            "component_tags": ",".join(sorted(component_tags)),
            "profile_text": profile_text,
            "profile_version": self.PROFILE_VERSION,
            "profile_sentence_count": len(final_sentences),
            "profile_char_count": len(profile_text),
        }

    def generate_etf_profiles(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Build one ETF profile record per ticker from sentence-level CSVs.
        """
        if not self.etf_doc_path.exists():
            logger.warning("ETF documentation path does not exist; skip profile generation.")
            return None

        target_output = output_path or (_PROJECT_ROOT / self.PROFILE_OUTPUT_RELATIVE_PATH)
        target_output.parent.mkdir(parents=True, exist_ok=True)
        ticker_dirs = [path for path in self.etf_doc_path.iterdir() if path.is_dir()]

        records: List[Dict[str, str]] = []
        for ticker_dir in tqdm(ticker_dirs, desc="Building ETF profiles"):
            profile = self._build_ticker_profile(ticker_dir)
            if profile:
                records.append(profile)

        if not records:
            logger.warning("No ETF profiles were generated.")
            return None

        df_profiles = pd.DataFrame(records).sort_values("ticker").reset_index(drop=True)
        df_profiles.to_csv(target_output, index=False, encoding="utf-8-sig")
        logger.info("Generated ETF profiles: %s (%d rows)", target_output, len(df_profiles))
        return target_output

    def _collect_pdf_jobs(self) -> List[Tuple[str, Path]]:
        """
        Build deduplicated per-ticker PDF jobs using existing version-key logic.
        """
        jobs: List[Tuple[str, Path]] = []
        etf_folders = [f for f in self.etf_doc_path.iterdir() if f.is_dir()]
        for folder in etf_folders:
            pdf_subdir = folder / "pdf"
            if not pdf_subdir.exists():
                continue

            pdf_files = sorted(list(pdf_subdir.glob("*.pdf")))
            processed_version_keys: Set[str] = set()
            for pdf_file in pdf_files:
                try:
                    version_key = pdf_file.stem.split("_")[-1]
                except IndexError:
                    version_key = pdf_file.stem
                if version_key in processed_version_keys:
                    continue
                jobs.append((folder.name, pdf_file))
                processed_version_keys.add(version_key)
        return jobs

    def extract_and_clean(self, pdf_path: Path) -> Optional[Path]:
        """
        Extracts text from a single PDF, applies the cleaning pipeline,
        and saves to a peer 'csv' directory.

        Args:
            pdf_path: Path object pointing to the source PDF.

        Returns:
            Path to the generated CSV if successful, else None.
        """
        # Define output path: ../pdf/file.pdf -> ../csv/file.csv
        output_dir = pdf_path.parent.parent / "csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_output_path = output_dir / pdf_path.with_suffix(".csv").name

        raw_content = []
        try:
            reader = pypdf.PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    cleaned_page = self._clean_text_segment(page_text)
                    raw_content.append(cleaned_page)

            # Combine and split into initial segments (primitive split)
            full_text = " ".join(raw_content)
            # Split more aggressively so legal blocks and table-like dumps do not become one giant sentence.
            split_pattern = r"(?<=[.!?;])\s+|(?<=\))\s+(?=\([a-z]\))|(?<=•)\s+"
            initial_segments = [s.strip() for s in re.split(split_pattern, full_text) if len(s.strip()) > 5]

            # Apply logic to merge broken fragments
            final_sentences = self._merge_fragments(initial_segments)

            # Create DataFrame
            df_sentences = pd.DataFrame({"sentence_id": range(len(final_sentences)), "text": final_sentences})

            # Save with utf-8-sig for Excel compatibility in HK/Asia regions
            df_sentences.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
            logger.info(f"Processed: {pdf_path.name}")
            return csv_output_path

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            return None

    def run_pipeline(
        self,
        generate_profiles: bool = True,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Orchestrates the extraction over the specific directory structure.
        """
        if not self.etf_doc_path.exists():
            logger.warning("Target documentation path does not exist.")
            return

        jobs = self._collect_pdf_jobs()
        logger.info("Prepared %d PDF extraction jobs.", len(jobs))

        if parallel and jobs:
            worker_count = max_workers or min(32, (os.cpu_count() or 1) + 4)
            worker_count = max(1, worker_count)
            logger.info("Running extraction in parallel with %d worker(s).", worker_count)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(self.extract_and_clean, pdf_path): (
                        ticker,
                        pdf_path,
                    )
                    for ticker, pdf_path in jobs
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                    ticker, pdf_path = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(
                            "Unhandled parallel failure for ticker %s (%s): %s",
                            ticker,
                            pdf_path.name,
                            exc,
                        )
        else:
            logger.info("Running extraction sequentially.")
            for _, pdf_path in tqdm(jobs, desc="Processing PDFs"):
                self.extract_and_clean(pdf_path)

        if generate_profiles:
            self.generate_etf_profiles()
        logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ETF PDF text and build ETF profiles.")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel PDF extraction.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker threads for parallel extraction (default: auto).",
    )
    parser.add_argument(
        "--no-profiles",
        action="store_true",
        help="Skip ETF profile generation after extraction.",
    )
    args = parser.parse_args()

    # The script automatically calculates path based on your folder structure.
    processor = ETFPDFProcessor()
    processor.run_pipeline(
        generate_profiles=not args.no_profiles,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )
