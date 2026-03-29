# STAT 8017B Project 4: Financial Analysis Chatbot

## Group 4.1 Members

| Name           | Email                        |
|----------------|------------------------------|
| Chan Chak Chung  | u3664795@connect.hku.hk    |
| Hung Man Kit     | u3643841@connect.hku.hk    |
| Jiang Hanbo      | u3664916@connect.hku.hk    |
| Lau Hiu Yu       | lauhiuyu@connect.hku.hk    |
| Leung Ho Ning    | hollishn@connect.hku.hk    |
| Weng Youjia      | u3637269@connect.hku.hk    |

## Key Deadlines

- Week 10 Discussion (Mar 29): Show data collection/cleansing progress and initial analysis
- Week 14 Presentation (Apr 29): 8-min YouTube video, max 20 slides
- Week 15 Final Submission (May 3, 23:59): Python program + Report (max 7200 words)

---

## Current Status (updated Mar 30)

| Component | Status | Notes |
|---|---|---|
| Data collection (8 datasets, 3.3 GB) | Done | All downloaded via Kaggle API |
| Preprocessing (`01_preprocessing.ipynb`) | Done | All 8 datasets cleaned; ~2,766 tickers loaded; 200K complaints (stratified) |
| Classification (`02_classification.ipynb`) | Done | NB, SVM, SVM+PCA, Decision Tree, RF, MLP on complaints + sentiment |
| Regression (`03_regression.ipynb`) | Done | Linear Reg + RF on fund returns and stock price trends |
| Unsupervised (`04_unsupervised.ipynb`) | Done | PCA, K-Means, Apriori association rules, LSA topic extraction, descriptive stats |
| Chatbot integration (`05_chatbot.ipynb`) | Done | Intent classifier, 6 analysis modules, Ollama/Phi-3.5 LLM integration |
| Streamlit app (`app/chatbot_app.py`) | Done | 5 panels: Chat, Fund Explorer, Complaints, Prices, Sentiment |
| Ollama + Phi-3.5 | Installed | Model pulled and ready on local machine |
| Report (7200 words) | Not started | Due May 3 |
| YouTube presentation (8 min) | Not started | Due Apr 29 |

### What's Left

1. **Run all notebooks in order** (01-05) to generate processed data and trained models
2. **Test the Streamlit chatbot** end-to-end: `streamlit run app/chatbot_app.py`
3. **Draft report** — 6 members x 1200 words
4. **Record 8-min YouTube presentation** — max 20 slides

### Course Methods Coverage

| Method | Notebook | Course Reference |
|---|---|---|
| NLTK Text Pipeline (tokenize, stopwords, stemming) | NB 01 | Ch.2 |
| TF-IDF + Cosine Similarity | NB 02, 04, 05 | Ch.2 |
| LSA (TruncatedSVD) | NB 04 | Ch.2 |
| Naive Bayes | NB 02 | Ch.3 |
| Decision Tree | NB 02 | Ch.4 |
| Random Forest (Classifier + Regressor) | NB 02, 03 | Ch.4 |
| SVM (LinearSVC) | NB 02 | Ch.5 |
| SVM + PCA | NB 02 | Ch.5 |
| PCA | NB 04 | Ch.5 |
| Linear Regression | NB 03 | Ch.5 |
| Neural Network (MLPClassifier) | NB 02 | Lecture 8 |
| K-Means Clustering | NB 04 | Tutorial 5 |
| Association Rules (Apriori) | NB 04 | Tutorial 6 |
| Descriptive Stats + Correlation | NB 04 | Ch.2 |

---

## 1. Problem Statement

Retail investors and financial analysts need quick, data-driven answers about financial products (ETFs, mutual funds, stocks). Existing tools are either too complex for beginners or too shallow for meaningful analysis. We aim to build a **Financial Analysis Chatbot** that can:

- Answer user queries about financial products using a structured knowledge base
- Perform statistical analysis (descriptive, correlation, regression) on demand
- Classify and analyze customer complaints about financial products
- Provide sentiment analysis on financial text
- Predict fund/stock performance trends
- Present results through an interactive dashboard

---

## 2. Datasets

### 2.1 Alpha Insights US Funds (ETFs + Mutual Funds)
- **Source**: https://www.kaggle.com/datasets/willianoliveiragibin/alpha-insights-us-funds
- **Contents**: Fund names, categories, expense ratios, returns (YTD, 1yr, 3yr, 5yr, 10yr), fund family, total net assets
- **Use**: Descriptive analytics, correlation analysis, regression, fund comparison
- **Files**: `ETFs.csv`, `mutual_funds.csv`

### 2.2 Bank Customer Complaints
- **Source**: https://www.kaggle.com/datasets/taeefnajib/bank-customer-complaints
- **Contents**: Complaint text, product category, issue type, company, state, date
- **Use**: Text mining, topic classification (NB/SVM), keyword extraction (TF-IDF)

### 2.3 Financial Phrasebank
- **Source**: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- **Contents**: ~4800 finance sentences labeled positive/negative/neutral
- **Use**: Train and validate sentiment analysis classifier

### 2.4 Finance Data (Stocks)
- **Source**: https://www.kaggle.com/datasets/nitindatta/finance-data
- **Contents**: Stock prices, financial ratios, company information
- **Use**: Stock trend analysis, numerical prediction models

### 2.5 Financial Q&A - 10K (NEW - for chatbot knowledge base)
- **Source**: https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k
- **Contents**: Pre-built question-answer pairs extracted from 10-K financial filings
- **Use**: Directly populate chatbot knowledge base; train Q&A retrieval with TF-IDF + cosine similarity

### 2.6 S&P 500, ETF, FX & Crypto - Daily Updated (NEW - richer price data)
- **Source**: https://www.kaggle.com/datasets/benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated
- **Contents**: 8000+ assets, daily OHLCV, dividends, stock splits (data up to 2025)
- **Use**: Broader stock/ETF universe for trend analysis, prediction models, time series charts
- **Note**: Large file (>1GB). Can sample top assets by market cap or filter to S&P 500 / popular ETFs.

### 2.7 Financial News Market Events 2025 (NEW - for sentiment & news QA)
- **Source**: https://www.kaggle.com/datasets/pratyushpuri/financial-news-market-events-dataset-2025
- **Contents**: Recent financial news articles with event labels
- **Use**: Sentiment analysis on real news, news-based Q&A, text classification

### Optional / Nice-to-Have
- **CFPB Consumer Complaints (7M+ rows)**: https://www.consumerfinance.gov/data-research/consumer-complaints -- massive version of 2.2, sample if needed
- **FinSen Financial Sentiment Dataset**: https://www.kaggle.com/datasets/eaglewhl/finsen-financial-sentiment-dataset -- alternative/supplement to 2.3
- **Financial Chat Bot Competition**: https://www.kaggle.com/competitions/financial-chat-bot -- reference notebooks and starter code

---

## 3. Technical Architecture

### 3.1 System Overview

```
User Input (question)
    |
    v
[Intent Classifier] -- classifies what the user wants
    |
    +--> Descriptive Analytics Module  (stats, charts)
    +--> Correlation Analysis Module   (scatter plots, r-values)
    +--> Regression / Prediction Module (trained models)
    +--> Topic Classification Module   (complaint categories)
    +--> Sentiment Analysis Module     (pos/neg/neutral)
    +--> Knowledge Base Lookup         (Q&A retrieval)
    |
    v
[Response Generator] -- formats answer + visualization
    |
    v
[Streamlit Dashboard + Chat UI]
```

### 3.2 Mapping to Project Objectives

| Project Objective | Our Approach |
|---|---|
| **Obj 1**: Knowledge base construction | Extract keywords with TF-IDF from complaint text + structure ETF/stock metadata into searchable Q&A pairs (JSON/dict) |
| **Obj 2**: Topic classification | Train Naive Bayes / SVM on complaint dataset to classify product categories and issue types |
| **Obj 3**: Text analysis & retrieval | Use TF-IDF + cosine similarity to match user queries to knowledge base entries; retrieve most relevant financial data |
| **Obj 4**: Numerical + text analysis | Descriptive stats on fund data (mean, median, mode, histograms); sentiment classification on financial text using Financial Phrasebank |
| **Obj 5**: Prediction models | Linear regression / Random Forest to predict fund returns; simple Markov chain or rule-based next-question suggestion |
| **Obj 6**: Integrated dashboard + chatbot | Streamlit app with plotly charts + chat interface combining all modules |

---

## 4. Data Preprocessing Plan

### 4.1 ETF / Mutual Fund Data
- Handle missing values (drop or impute with median)
- Convert percentage strings to float
- Standardize category names
- Remove duplicate entries
- Feature engineering: return-to-expense ratio, risk-adjusted returns

### 4.2 Customer Complaints Text Data
- Lowercase all text
- Remove special characters, URLs, numbers
- Tokenize with NLTK
- Remove stopwords
- Apply stemming (PorterStemmer) or lemmatization
- TF-IDF vectorization for model input

### 4.3 Financial Phrasebank
- Split into train/test (80/20 stratified)
- Same text preprocessing as above
- TF-IDF vectorization

### 4.4 Stock / Finance Data
- Handle missing values
- Normalize numerical features (StandardScaler)
- Time-series formatting if applicable

---

## 5. Models & Methods

### 5.1 Descriptive Analytics
- Mean, median, mode, std dev of fund returns and expense ratios
- Histograms, box plots, bar charts by category
- Frequency distributions of complaint types

### 5.2 Correlation Analysis
- Pearson correlation: expense ratio vs. returns
- Correlation matrix heatmap across fund features
- Scatter plots with regression lines

### 5.3 Classification Models
- **Naive Bayes (MultinomialNB)**: baseline for topic classification and sentiment
- **SVM (LinearSVC)**: improved classifier for both tasks
- **SVM + PCA**: dimensionality reduction before SVM (mirrors Assignment 1 Ch.5)
- **Decision Tree**: interpretable classifier for comparison
- **Random Forest**: ensemble comparison
- **Neural Network (MLPClassifier)**: deep learning approach (matches Lecture 8)
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix, ROC/AUC

### 5.4 Regression / Prediction
- **Linear Regression**: predict returns from expense ratio, fund size, category
- **Random Forest Regressor**: capture non-linear relationships
- Feature importance analysis
- Evaluation: R-squared, MAE, RMSE

### 5.5 Text Analysis
- **TF-IDF**: keyword extraction and document vectorization
- **Cosine Similarity**: query-to-document matching for chatbot retrieval
- **LSA (Latent Semantic Analysis)**: TruncatedSVD on TF-IDF for latent topic extraction
- **Word frequency analysis**: most discussed topics in complaints

### 5.6 Sentiment Analysis
- Train on Financial Phrasebank (positive/negative/neutral)
- Apply to user-provided text or financial headlines
- Cross-dataset validation on FinSen articles
- Output: sentiment label + confidence score

### 5.7 Unsupervised Learning
- **PCA**: dimensionality reduction on fund features, explained variance analysis, 2D/3D visualization
- **K-Means Clustering**: segment funds by performance profile (elbow method + silhouette score)
- **Association Rules (Apriori)**: mine co-occurrence patterns in complaint attributes (Product, Issue, State)

---

## 6. Chatbot Conversation Design

Example interactions the chatbot should handle:

```
User: "What are the top performing ETFs?"
Bot:  [Queries ETF data, sorts by return, shows top 10 table + bar chart]

User: "Is expense ratio correlated with returns?"
Bot:  [Runs Pearson correlation, shows scatter plot + r-value + interpretation]

User: "Predict return for a tech ETF with 0.5% expense ratio"
Bot:  [Loads regression model, predicts, returns estimate with confidence]

User: "What do customers complain about most?"
Bot:  [Shows top complaint categories with frequency chart]

User: "Analyze sentiment: 'The market showed strong gains today'"
Bot:  [Runs sentiment classifier, returns: Positive (confidence: 0.87)]

User: "Compare ETFs vs Mutual Funds"
Bot:  [Shows side-by-side descriptive stats, box plots of returns]
```

---

## 7. Tech Stack

| Component | Library |
|---|---|
| Data processing | pandas, numpy |
| Text preprocessing | nltk, re |
| ML models | scikit-learn (NB, SVM, DT, RF, MLP, LinearRegression, PCA, K-Means) |
| TF-IDF & similarity | scikit-learn (TfidfVectorizer, cosine_similarity, TruncatedSVD) |
| Association Rules | mlxtend (Apriori, association_rules) |
| Visualization | matplotlib, plotly |
| Dashboard + Chat UI | streamlit |
| LLM | ollama (Phi-3.5 Mini) |
| Model persistence | joblib |

---

## 8. Project File Structure

```
8017project/
├── data/
│   ├── alpha-insights/          # Raw ETF + Mutual Fund data
│   ├── complaints/              # Raw customer complaints
│   ├── finance-data/            # Raw investment survey
│   ├── financial-news/          # Raw financial news
│   ├── financial-qa/            # Raw Q&A pairs from 10-K filings
│   ├── finsen/                  # Raw FinSen financial sentiment
│   ├── phrasebank/              # Raw Financial Phrasebank
│   ├── sp500-etf-crypto/        # Raw price CSVs (~2,766 tickers)
│   └── processed/               # All cleaned data (CSVs + joblib artifacts)
├── notebooks/
│   ├── 01_preprocessing.ipynb   # Data preprocessing (all 8 datasets)
│   ├── 02_classification.ipynb  # NB, SVM, DT, RF, MLP classifiers
│   ├── 03_regression.ipynb      # Linear Reg, RF Regressor
│   ├── 04_unsupervised.ipynb    # PCA, K-Means, Apriori, LSA, descriptive stats
│   └── 05_chatbot.ipynb         # Chatbot integration + LLM
├── models/                      # Trained model .joblib files
├── app/
│   └── chatbot_app.py           # Streamlit chatbot + dashboard (5 panels)
├── project_plan.md              # This file
├── README.md                    # Setup instructions for teammates
├── requirements.txt
└── .gitignore
```

---

## 9. Work Division Suggestion (6 members)

| Member | Responsibility | Report Section |
|---|---|---|
| Member 1 | Data collection, cleaning, preprocessing | Ch 3.1 + part of Ch 4 |
| Member 2 | Descriptive analytics & correlation analysis | Ch 3.2 + part of Ch 4 |
| Member 3 | Topic classification (complaints) | Ch 2 (lit review) + part of Ch 4 |
| Member 4 | Sentiment analysis | Ch 2 (lit review) + part of Ch 4 |
| Member 5 | Regression / prediction models | Ch 3.2 + part of Ch 4 |
| Member 6 | Chatbot integration, dashboard, UI | Ch 2 (lit review) + part of Ch 4 |
| All | Ch 1 (Introduction), Ch 5 (Discussion), Ch 6 (Conclusion) | Group sections |

---

## 10. Timeline

| Week | Milestone |
|---|---|
| Week 10 (Mar 29) | Datasets downloaded, initial EDA complete, show to Dr. Lau |
| Week 11 (Apr 5) | Data preprocessing done, first models trained |
| Week 12 (Apr 12) | All models trained and evaluated, knowledge base built |
| Week 13 (Apr 19) | Chatbot integrated, dashboard working, report draft |
| Week 14 (Apr 29) | YouTube presentation recorded and uploaded |
| Week 15 (May 3) | Final program + report submitted |
