import pypdf
import re
import pandas as pd
import os
from pathlib import Path

def extract_to_specific_csv(pdf_path):
    # 1. Define the output path based on your requirement
    # Logic: Go up one level from 'pdf/', create 'csv/' folder, and change extension
    pdf_file = Path(pdf_path)
    output_dir = pdf_file.parent.parent / "csv"
    output_csv_path = output_dir / pdf_file.with_suffix(".csv").name

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    raw_pages = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Clean dot leaders (....) to prevent empty string splits
                text = re.sub(r'\.{2,}', ' ', text)
                raw_pages.append(text)
        
        full_text = "\n\n".join(raw_pages)
        
        # Split into sentences and filter for meaningful content
        sentences = [s.strip() for s in full_text.split(".") if len(s.strip()) > 5]
        
        # 2. Create DataFrame and save
        df = pd.DataFrame({
            "sentence_id": range(len(sentences)),
            "text": sentences
        })
        
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Successfully converted:\n{pdf_path}\n->\n{output_csv_path}")
        
        return output_csv_path

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

# Execution
pdf_input = "/Users/hunglun/Library/Mobile Documents/com~apple~CloudDocs/Study/HKU_STAT/2025_Sem2/STAT8020/Project/8017Group4.1/data/ETF/documentation/02800/pdf/1776534841_9_Prospectus.pdf"
extract_to_specific_csv(pdf_input)