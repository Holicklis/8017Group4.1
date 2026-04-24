import os
import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm

class ETFNewsEngine:
    def __init__(self, csv_dir: str, metadata_csv: str, cache_path: str = "corpus_embeddings.pt"):
        """
        Args:
            csv_dir: Path to 'data/etf/documentation' (the folder containing ticker subfolders)
            metadata_csv: Path to your ETP_Data_Export csv
        """
        # 1. Device Selection (Optimized for M2 Pro)
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing Engine on: {self.device}")

        # 2. Load Metadata and build lookup
        self.metadata_df = pd.read_excel(metadata_csv).head(-2)
        self.metadata_df['Stock code*'] = self.metadata_df['Stock code*'].astype(int).astype(str).str.strip("0")
        self._prepare_metadata_cache()

        # 3. Load Documentation from Nested Folders
        self.docs_df = self._load_all_csvs(csv_dir)
        
        # 4. Load AI Models
        print("Loading NLP Models...")
        self.bi_encoder = SentenceTransformer('BAAI/bge-m3', device=self.device)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

        # 5. Handle Caching for Embeddings (The time-consuming part)
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}...")
            self.corpus_embeddings = torch.load(cache_path, map_location=self.device)
        else:
            print(f"Encoding {len(self.docs_df)} chunks (Running on {self.device})...")
            self.corpus_embeddings = self.bi_encoder.encode(
                self.docs_df['text'].tolist(), 
                convert_to_tensor=True, 
                show_progress_bar=True,
                batch_size=16 
            )
            torch.save(self.corpus_embeddings, cache_path)
            print(f"Saved embeddings to {cache_path}")

    def _load_all_csvs(self, csv_dir: str) -> pd.DataFrame:
        """Traverses ticker folders (02800, 02801, etc.) and loads all CSVs."""
        all_dfs = []
        base_path = Path(csv_dir)
        ticker_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        
        print(f"Found {len(ticker_dirs)} ETF folders. Loading documentation...")
        for ticker_dir in tqdm(ticker_dirs, desc="Reading CSVs"):
            ticker_id = ticker_dir.name # e.g., '02800'
            
            for csv_file in ticker_dir.glob("csv/*.csv"):
                if "product key fact" in csv_file.name.lower():
                    try:
                        temp_df = pd.read_csv(csv_file)
                        if 'text' in temp_df.columns:
                            # Extract only what we need to save memory
                            subset = temp_df[['text']].copy()
                            subset['ticker'] = ticker_id
                            all_dfs.append(subset)
                        break
                    except Exception as e:
                        print(f"Could not load {csv_file}: {e}")


        return pd.concat(all_dfs, ignore_index=True)

    def _prepare_metadata_cache(self):
        """Creates a lookup dictionary. Keyed by both string and float for robustness."""
        self.meta_lookup = {}
        for _, row in self.metadata_df.iterrows():
            raw_code = row['Stock code*']
            # Store by numeric float (2800.0) AND string ('02800' and '2800')
            meta_data = row.to_dict()
            self.meta_lookup[raw_code] = meta_data
            try:
                # Add '2800' (string)
                self.meta_lookup[str(int(float(raw_code)))] = meta_data
                # Add '02800' (string)
                self.meta_lookup[str(int(float(raw_code))).zfill(5)] = meta_data
            except:
                pass

    def _calculate_boost(self, news_text: str, row: pd.Series) -> float:
        """Applies extra weight if the news matches the ticker, asset class, or domicile."""
        boost = 0.0
        news_lower = news_text.lower()
        ticker = row['ticker'] # Folder name like '02800'
        
        # 1. Direct Ticker Match (Clean '2800' and Original '02800')
        clean_ticker = ticker.lstrip('0')
        if clean_ticker in news_lower or ticker in news_lower:
            boost += 0.8

        # 2. Category Match (Asset Class, Focus, Domicile)
        meta = self.meta_lookup.get(ticker)
        if meta:
            # Add boost for metadata fields defined by the user
            for field in ['Asset class*', 'Investment focus*', 'Country of domicile*', 'Geographic focus*']:
                val = str(meta.get(field, "")).lower()
                if val != "nan" and val in news_lower:
                    boost += 0.25
                    
        return boost

    def search(self, news_headline: str, top_k: int = 5):
        """Standard Search Pipeline with M2 Pro Acceleration."""
        # --- Stage 1: Retrieval ---
        query_emb = self.bi_encoder.encode(news_headline, convert_to_tensor=True)
        # We get 2x top_k to allow re-ranking to bubble up boosted matches
        hits = util.semantic_search(query_emb, self.corpus_embeddings, top_k=top_k*2)[0]
        
        results = []
        for hit in hits:
            row = self.docs_df.iloc[hit['corpus_id']]
            
            # --- Stage 2: Deep Re-ranking ---
            semantic_score = self.cross_encoder.predict([news_headline, row['text']])
            
            # --- Stage 3: Metadata Boosting ---
            meta_boost = self._calculate_boost(news_headline, row)
            
            results.append({
                'ticker': row['ticker'],
                'text': row['text'],
                'final_score': float(semantic_score) + meta_boost,
                'boost': round(meta_boost, 2)
            })
            
        # Sort results by final score
        sorted_results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        return sorted_results[:top_k]

# --- EXECUTION ---
# Change paths to match your local setup
_data_root = Path(__file__).resolve().parents[3] / "data"
_etf_root = _data_root / "etf" if (_data_root / "etf").exists() else _data_root / "ETF"
_summary_dir = _etf_root / "summary" if (_etf_root / "summary").exists() else _etf_root / "Summary"

CSV_ROOT = str(_etf_root / "documentation")
META_PATH = str(_summary_dir / "ETP_Data_Export.xlsx")

engine = ETFNewsEngine(csv_dir=CSV_ROOT, metadata_csv=META_PATH)

# Test run
headline = "HKMA interest rate changes affect Hong Kong Money Market ETFs"
results = engine.search(headline)

for res in results:
    print(f"\n[{res['ticker']}] Score: {res['final_score']:.2f} (Boost: {res['boost']})")
    print(f"Snippet: {res['text'][:200]}...")