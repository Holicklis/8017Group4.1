import os
import re
import logging
import pandas as pd
import pypdf
from pathlib import Path
from typing import List, Optional, Union
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ETFPDFProcessor")

class ETFPDFProcessor:
    """
    A professional pipeline to extract, clean, and structure ETF prospectus data 
    from PDF files into CSV format for LLM fine-tuning and vector embeddings.
    """

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
            # Script is in src/text_extract/, data is in ../../data/
            self.data_dir = Path(__file__).resolve().parents[2] / "data"
        
        self.etf_doc_path = self.data_dir / "ETF" / "documentation"
        
        if not self.data_dir.exists():
            logger.error(f"Data directory not found at: {self.data_dir}")
        else:
            logger.info(f"Initialized ETFPDFProcessor. Data Root: {self.data_dir}")

    def _clean_text_segment(self, text: str) -> str:
        """
        Standard text data cleaning: removes noise, artifacts, and formatting issues.
        """
        if not text:
            return ""
        
        # 1. Remove dot leaders (....) common in Tables of Contents
        text = re.sub(r'\.{2,}', ' ', text)
        
        # 2. Remove Page Indicators (e.g., "- i -", "- 1 -", "- 22 -")
        text = re.sub(r'-\s*[ivx0-9]+\s*-', '', text, flags=re.IGNORECASE)
        
        # 3. Standardize whitespace (remove newlines/tabs)
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        # 4. Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _merge_fragments(self, sentences: List[str]) -> List[str]:
        """
        Heuristic to rejoin sentences split by PDF line/page breaks.
        """
        merged = []
        current_buffer = ""
        
        terminal_punctuation = {'.', '!', '?', '”', '"'}

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
            # Use regex split to keep separators or split by common sentence ends
            initial_segments = [s.strip() for s in re.split(r'(?<=[.!?]) +', full_text) if len(s.strip()) > 5]
            
            # Apply logic to merge broken fragments
            final_sentences = self._merge_fragments(initial_segments)

            # Create DataFrame
            df = pd.DataFrame({
                "sentence_id": range(len(final_sentences)),
                "text": final_sentences
            })

            # Save with utf-8-sig for Excel compatibility in HK/Asia regions
            df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Processed: {pdf_path.name}")
            return csv_output_path

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            return None

    def run_pipeline(self):
        """
        Orchestrates the extraction over the specific directory structure.
        """
        if not self.etf_doc_path.exists():
            logger.warning("Target documentation path does not exist.")
            return

        # Locate all subfolders in ETF/documentation
        etf_folders = [f for f in self.etf_doc_path.iterdir() if f.is_dir()]
        
        logger.info(f"Found {len(etf_folders)} ETF folders. Starting extraction...")

        for folder in tqdm(etf_folders, desc="Processing ETFs"):
            pdf_subdir = folder / "pdf"
            if not pdf_subdir.exists():
                continue

            # Sort and find unique versions (mimicking your 'exist' logic)
            pdf_files = sorted(list(pdf_subdir.glob("*.pdf")))
            processed_version_keys = set()

            for pdf_file in pdf_files:
                # Key extraction logic from your script
                try:
                    version_key = pdf_file.stem.split("_")[-1]
                except IndexError:
                    version_key = pdf_file.stem

                if version_key not in processed_version_keys:
                    self.extract_and_clean(pdf_file)
                    processed_version_keys.add(version_key)

        logger.info("Pipeline execution complete.")

if __name__ == "__main__":
    # The script automatically calculates path based on your folder structure
    processor = ETFPDFProcessor()
    processor.run_pipeline()