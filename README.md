# HK ETF Global Navigator 🚀
### An AI-Powered Platform to Mitigate Home Bias in HK Retail Investing

## Project Overview

HK ETF Global Navigator is an intelligent investment assistant designed to help Hong Kong retail investors diversify beyond domestic markets.
It combines structured financial analytics with NLP-driven insights to provide:

- ETF classification and ranking
- Context-aware financial news matching
- A factual AI advisory chatbot grounded in trusted sources

The goal is to turn fragmented market information into actionable, explainable guidance for everyday investors.

---

## Why This Project Matters

Many HK retail investors exhibit home bias - over-concentrating portfolios in local equities and missing global diversification opportunities.
This platform addresses that gap by combining machine learning, semantic search, and retrieval-augmented chat to support better decision-making.

---

## Core Architecture

| Task | Proposed Model | Why This Model |
|---|---|---|
| ETF Classification | Random Forest / XGBoost | Handles non-linear financial relationships (expense ratio, volatility, AUM) better than basic linear models |
| News Recommendation | Sentence-BERT (SBERT) | Captures semantic meaning, not just keyword overlap |
| Chatbot Engine | RAG (Retrieval-Augmented Generation) | Grounds responses in trusted documents and reduces hallucinations |

---

## Data Strategy

To ensure transparency and analytical depth, the pipeline integrates three data layers:

### 1) Structured Financial Data
- HKEX ETF metadata
- Fund attributes (fees, asset class, dividend policy)
- Portfolio-level features for modeling

### 2) Market Time-Series Data
- Yahoo Finance historical prices
- Return engineering and volatility features
- True Total Return adjusted for dividend distributions

### 3) Unstructured Text Data
- Financial news (Kaggle / Reuters)
- ETF prospectus and policy documents (PDF)
- Text corpus indexed for semantic retrieval

---

## End-to-End Pipeline

1. **Data Ingestion**
   Collect and standardize structured + unstructured sources

2. **Feature Engineering**
   Compute return/risk features, normalize metadata, build text embeddings

3. **Modeling**
   Train classifiers/rankers and semantic retrieval components

4. **Knowledge Retrieval**
   Build vector index for ETF documents and macro news context

5. **Advisory Generation**
   Chatbot routes user intent, retrieves evidence, and produces reasoned recommendations

---

## Integrated Dashboard

The final deliverable is an interactive Python dashboard with two major modules:

### 1) Interactive Screener
- Cross-asset performance comparison (line and ranking charts)
- Theme leaderboard (e.g., Tech vs Energy vs Gold)
- ETF filtering by risk, cost, and market exposure

### 2) AI Advisory Chatbot
A 3-step reasoning workflow:

- **User Input**
  Example: "I am worried about inflation and want gold exposure."

- **Intent Classification**
  Detects profile/theme (e.g., conservative hedge-oriented intent)

- **Action + Recommendation**
  Retrieves relevant HK-listed gold ETFs + latest macro context and returns an explainable recommendation

---

## Repository Structure

```text
├── data/               # Raw and cleaned datasets
├── models/             # Trained model weights and artifacts
├── notebooks/          # EDA, prototyping, and experiment workflows
├── src/                # Core application and pipeline source code
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd hk-etf-global-navigator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
python app.py
```

---

## Example User Questions

- "Which ETFs best hedge inflation risk right now?"
- "Show low-cost global ETFs with stable historical returns."
- "Compare gold ETF performance against tech ETFs in the past 12 months."
- "What macro news might affect energy-themed ETFs this week?"

---

## Future Enhancements

- Real-time market data streaming
- Portfolio optimization suggestions
- User risk profiling and personalization
- Backtesting module for strategy comparison

---

## Disclaimer

This platform is for educational and research demonstration purposes only and does not constitute financial advice.