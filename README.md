Got it. I have merged your original project narrative with a high-powered **"Technical Implementation & Optimization"** section. This allows the reader to first understand the *business value* (your original content) and then see the *engineering depth* (the AI and M2 hardware work we just did).

Here is the full, improved `README.md`.

---

# HK ETF Global Navigator 🚀
### An AI-Powered Platform to Mitigate Home Bias in HK Retail Investing

## Project Overview
HK ETF Global Navigator is an intelligent investment assistant designed to help Hong Kong retail investors diversify beyond domestic markets. It combines structured financial analytics with NLP-driven insights to provide:

- **ETF Classification and Ranking:** Data-driven categorization of fund performance.
- **Context-aware Financial News Matching:** NLP matching between global news and HK-listed ETFs.
- **Factual AI Advisory Chatbot:** Grounded responses using official documentation to reduce hallucinations.

The goal is to turn fragmented market information into actionable, explainable guidance for everyday investors.

---

## 🛠 Technical Implementation & Optimization
To handle the scale of **500,000+ document chunks**, the platform uses a sophisticated retrieval architecture specifically optimized for **Apple Silicon**.

### 1. Two-Stage "Retrieve & Re-rank" Pipeline
Traditional keyword search is insufficient for dense financial text. This engine uses a dual-model approach:
* **Stage 1: Semantic Retrieval (Bi-Encoder):** Uses `BAAI/bge-m3` to embed the entire corpus. This allows the system to understand that a news story about "Interest Rate Hikes" is relevant to "Money Market ETFs," even if the words don't match exactly.
* **Stage 2: Contextual Re-ranking (Cross-Encoder):** The top candidates are re-scored by a `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`). This model performs a deep comparison of the news headline against the document text to ensure precise relevance.

### 2. Multi-Factor Metadata Boosting
The engine doesn't just rely on AI; it integrates a **Financial Logic Layer**:
* **Ticker Recognition:** Direct boosting when stock codes (e.g., `02800`) are detected in headlines.
* **Thematic Alignment:** Dynamic weight increases if news content matches an ETF's **Asset Class**, **Investment Focus**, or **Geographic Focus** defined in the HKEX metadata.

### 3. Hardware Acceleration (M2 Pro)
Processing 560k chunks is computationally expensive. The system is engineered to maximize **M2 Pro** performance:
* **MPS (Metal Performance Shaders):** Computation is offloaded to the Mac's GPU cores via `torch.backends.mps`, providing a 5-10x speedup over CPU-only processing.
* **Persistent Vector Cache:** The high-cost embedding process is performed once and serialized to a `.pt` file, allowing the engine to initialize instantly in future sessions.

---

## Why This Project Matters
Many HK retail investors exhibit home bias—over-concentrating portfolios in local equities and missing global diversification opportunities. This platform addresses that gap by combining machine learning, semantic search, and retrieval-augmented chat to support better decision-making.

---

## Core Architecture

| Task | Proposed Model | Why This Model |
|---|---|---|
| **ETF Classification** | Random Forest / XGBoost | Handles non-linear financial relationships better than basic linear models. |
| **News Recommendation** | Bi-Encoder + Cross-Encoder | Balances retrieval speed with deep semantic accuracy. |
| **Hardware Backend** | Apple Silicon (MPS) | Utilizes GPU cores of M-series chips for high-speed AI processing. |
| **Chatbot Engine** | RAG (Retrieval-Augmented) | Grounds responses in trusted PDF documents to eliminate AI hallucinations. |

---

## Data Strategy
To ensure transparency and analytical depth, the pipeline integrates three data layers:

### 1) Structured Financial Data
- HKEX ETF metadata (fees, asset class, dividend policy).
- Master lookup tables linking folder-based tickers to numeric stock codes.

### 2) Market Time-Series Data
- Yahoo Finance historical prices and volatility features.
- True Total Return adjusted for dividend distributions.

### 3) Unstructured Text Data
- **Documentation:** Recursive ingestion of 560k+ text chunks from official ETF prospectuses partitioned by ticker folders.
- **Market News:** Contextual news corpus indexed for semantic retrieval.

---

## End-to-End Pipeline
1.  **Data Ingestion:** Collect and standardize structured + unstructured sources across nested directories.
2.  **Feature Engineering:** Compute risk/return features and generate 1024-dimensional text embeddings.
3.  **Indexing:** Build a persistent vector index for the half-million chunk documentation corpus.
4.  **Actionable Retrieval:** Match user intent or news context to specific ETFs using the **Boosted AI Search**.

---

## Repository Structure
```text
├── data/               
│   ├── ETF/
│   │   ├── documentation/     # Nested folders by Ticker (02800, 02801...)
│   │   └── ETP_Metadata.csv   # Master ETF attributes
├── models/             
│   └── corpus_embeddings.pt   # Cached M2 Pro GPU embeddings
├── notebooks/          # EDA and prototyping workflows
├── src/                # Core application source code
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## Getting Started
### 1. Install Dependencies
Ensure you are using a native **arm64** Python environment (3.11+ recommended).
```bash
pip install torch torchvision torchaudio sentence-transformers pandas tqdm
```

### 2. Run the Dashboard
```bash
python app.py
```

---

## Disclaimer
This platform is for educational and research demonstration purposes only and does not constitute financial advice.