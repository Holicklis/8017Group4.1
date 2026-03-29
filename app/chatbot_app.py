"""
STAT 8017B Project 4 — Financial Analysis Chatbot
Streamlit Application (Group 4.1)

Run: streamlit run app/chatbot_app.py
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(stemmer.stem(t) for t in tokens)


# ---------------------------------------------------------------------------
# Load data & models (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    res = {}
    res["etf"] = pd.read_csv(os.path.join(PROCESSED_DIR, "etf_clean.csv"))
    res["mf"] = pd.read_csv(os.path.join(PROCESSED_DIR, "mutualfund_clean.csv"))
    res["comp"] = pd.read_csv(os.path.join(PROCESSED_DIR, "complaints_clean.csv"))
    res["qa"] = pd.read_csv(os.path.join(PROCESSED_DIR, "qa_clean.csv"))
    res["prices"] = pd.read_csv(os.path.join(PROCESSED_DIR, "prices_clean.csv"))
    res["news"] = pd.read_csv(os.path.join(PROCESSED_DIR, "news_clean.csv"))

    res["qa_tfidf"] = joblib.load(os.path.join(PROCESSED_DIR, "qa_tfidf_vectorizer.joblib"))
    res["qa_matrix"] = joblib.load(os.path.join(PROCESSED_DIR, "qa_tfidf_matrix.joblib"))

    res["comp_tfidf"] = joblib.load(os.path.join(PROCESSED_DIR, "complaints_tfidf.joblib"))
    res["comp_labels"] = joblib.load(os.path.join(PROCESSED_DIR, "complaints_labels.joblib"))

    res["sent_le"] = joblib.load(os.path.join(PROCESSED_DIR, "phrasebank_label_encoder.joblib"))

    for name in [
        "complaint_svm", "sentiment_svm", "sentiment_tfidf",
        "fund_return_lr", "fund_return_rf", "fund_return_scaler",
        "intent_tfidf", "intent_matrix", "intent_labels",
    ]:
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        if os.path.exists(path):
            res[name] = joblib.load(path)

    return res


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------
def classify_intent(query, res, threshold=0.15):
    if "intent_tfidf" not in res:
        return "general_qa", 0.0
    vec = res["intent_tfidf"].transform([clean_text(query)])
    sims = cosine_similarity(vec, res["intent_matrix"]).flatten()
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    if best_score < threshold:
        return "general_qa", best_score
    return res["intent_labels"][best_idx], float(best_score)


# ---------------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------------
def module_fund_lookup(query, res):
    etf = res["etf"]
    if "fund_return_1year" in etf.columns:
        top = etf.nlargest(10, "fund_return_1year")
        cols = ["fund_symbol", "fund_short_name", "fund_category",
                "fund_return_1year", "fund_annual_report_net_expense_ratio"]
        cols = [c for c in cols if c in top.columns]
        return "Top 10 ETFs by 1-Year Return", top[cols], None
    return "No return data available.", pd.DataFrame(), None


def module_complaint_analysis(query, res):
    dist = res["comp"]["Product"].value_counts().head(10)
    fig = px.bar(x=dist.index, y=dist.values, labels={"x": "Product", "y": "Count"},
                 title="Top 10 Complaint Categories")
    return "Top 10 complaint product categories", dist.reset_index(), fig


def module_sentiment(query, res):
    if "sentiment_svm" not in res or "sentiment_tfidf" not in res:
        return "Sentiment model not loaded.", pd.DataFrame(), None
    vec = res["sentiment_tfidf"].transform([clean_text(query)])
    pred = res["sentiment_svm"].predict(vec)[0]
    label = res["sent_le"].inverse_transform([pred])[0]
    scores = res["sentiment_svm"].decision_function(vec)[0]
    score_dict = {res["sent_le"].inverse_transform([i])[0]: round(float(s), 4)
                  for i, s in enumerate(scores)}
    df = pd.DataFrame([score_dict])
    return f"Sentiment: **{label}**", df, None


def module_prediction(query, res):
    if "fund_return_lr" not in res:
        return "Prediction model not loaded.", pd.DataFrame(), None
    etf = res["etf"]
    feature_cols = [
        "fund_annual_report_net_expense_ratio", "total_net_assets", "fund_yield",
        "fund_return_ytd", "fund_return_3years", "fund_return_5years",
        "fund_mean_annual_return_3years", "fund_stdev_3years",
        "fund_sharpe_ratio_3years", "fund_beta_3years",
        "asset_stocks", "asset_bonds",
    ]
    available = [c for c in feature_cols if c in etf.columns]
    median_vals = etf[available].median().values.reshape(1, -1)
    scaled = res["fund_return_scaler"].transform(median_vals)
    pred_lr = res["fund_return_lr"].predict(scaled)[0]
    pred_rf = res["fund_return_rf"].predict(scaled)[0]
    df = pd.DataFrame([{
        "Model": "Linear Regression", "Predicted 1yr Return": round(pred_lr, 4)
    }, {
        "Model": "Random Forest", "Predicted 1yr Return": round(pred_rf, 4)
    }])
    return "Predicted 1-year return (median fund profile)", df, None


def module_price_trend(query, res):
    prices = res["prices"]
    tickers = prices["ticker"].unique()
    query_upper = query.upper()
    found = None
    for t in tickers:
        if t.upper() in query_upper:
            found = t
            break
    if found is None:
        found = tickers[0] if len(tickers) > 0 else None
    if found is None:
        return "No price data available.", pd.DataFrame(), None

    ticker_df = prices[prices["ticker"] == found].copy()
    ticker_df["Date"] = pd.to_datetime(ticker_df["Date"])
    ticker_df = ticker_df.sort_values("Date").tail(252)
    price_col = "Close" if "Close" in ticker_df.columns else "Adj Close"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticker_df["Date"], y=ticker_df[price_col], name="Close"))
    if "ma_20" in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df["Date"], y=ticker_df["ma_20"],
                                 name="MA 20", line=dict(dash="dash")))
    if "ma_50" in ticker_df.columns:
        fig.add_trace(go.Scatter(x=ticker_df["Date"], y=ticker_df["ma_50"],
                                 name="MA 50", line=dict(dash="dot")))
    fig.update_layout(title=f"{found} Price Trend", xaxis_title="Date", yaxis_title="Price")
    return f"Price trend for {found}", ticker_df.tail(30), fig


def module_general_qa(query, res):
    vec = res["qa_tfidf"].transform([clean_text(query)])
    sims = cosine_similarity(vec, res["qa_matrix"]).flatten()
    top_idx = sims.argsort()[-3:][::-1]
    qa = res["qa"]
    results = []
    for idx in top_idx:
        results.append({
            "Score": round(float(sims[idx]), 4),
            "Question": qa.iloc[idx]["question"],
            "Answer": qa.iloc[idx]["answer"][:300],
        })
    return "Retrieved from Financial Q&A Knowledge Base", pd.DataFrame(results), None


MODULE_MAP = {
    "fund_lookup": module_fund_lookup,
    "complaint_analysis": module_complaint_analysis,
    "sentiment": module_sentiment,
    "prediction": module_prediction,
    "price_trend": module_price_trend,
    "general_qa": module_general_qa,
}


# ---------------------------------------------------------------------------
# LLM response (Ollama / fallback)
# ---------------------------------------------------------------------------
def llm_response(query, summary, data_context):
    try:
        from ollama import chat
        system = (
            "You are a Financial Analysis Chatbot. Summarize the analysis results "
            "in clear natural language for a retail investor. Keep it under 200 words."
        )
        user_msg = f"User question: {query}\n\nAnalysis summary: {summary}\n\nData:\n{data_context}"
        resp = chat(model="phi3.5", messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])
        return resp["message"]["content"]
    except Exception:
        return summary


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Financial Analysis Chatbot", page_icon="📊", layout="wide")

    st.title("Financial Analysis Chatbot")
    st.caption("STAT 8017B Project 4 — Group 4.1")

    res = load_resources()

    # Sidebar
    with st.sidebar:
        st.header("Dashboard")
        panel = st.radio("Select View", [
            "Chat", "Fund Explorer", "Complaint Dashboard",
            "Price Charts", "Sentiment Demo",
        ])

    if panel == "Chat":
        render_chat(res)
    elif panel == "Fund Explorer":
        render_fund_explorer(res)
    elif panel == "Complaint Dashboard":
        render_complaint_dashboard(res)
    elif panel == "Price Charts":
        render_price_charts(res)
    elif panel == "Sentiment Demo":
        render_sentiment_demo(res)


def render_chat(res):
    st.subheader("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("df") is not None:
                st.dataframe(msg["df"], use_container_width=True)
            if msg.get("fig") is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)

    if prompt := st.chat_input("Ask about ETFs, complaints, sentiment, predictions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            intent, score = classify_intent(prompt, res)
            st.caption(f"Intent: `{intent}` (score: {score:.3f})")

            module_fn = MODULE_MAP.get(intent, module_general_qa)
            summary, df, fig = module_fn(prompt, res)

            data_str = df.head(10).to_string(index=False) if not df.empty else ""
            response_text = llm_response(prompt, summary, data_str)

            st.markdown(response_text)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        st.session_state.messages.append({
            "role": "assistant", "content": response_text,
            "df": df if not df.empty else None,
            "fig": fig,
        })


def render_fund_explorer(res):
    st.subheader("Fund Explorer")
    etf = res["etf"]

    if "fund_category" in etf.columns:
        categories = ["All"] + sorted(etf["fund_category"].dropna().unique().tolist())
        selected = st.selectbox("Filter by Category", categories)
        if selected != "All":
            etf = etf[etf["fund_category"] == selected]

    st.metric("Total Funds", len(etf))

    if "fund_return_1year" in etf.columns:
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg 1yr Return", f"{etf['fund_return_1year'].mean():.2f}%")
        col2.metric("Max 1yr Return", f"{etf['fund_return_1year'].max():.2f}%")
        if "fund_annual_report_net_expense_ratio" in etf.columns:
            col3.metric("Avg Expense Ratio",
                        f"{etf['fund_annual_report_net_expense_ratio'].mean():.4f}")

        fig = px.scatter(
            etf.dropna(subset=["fund_return_1year", "fund_annual_report_net_expense_ratio"]),
            x="fund_annual_report_net_expense_ratio", y="fund_return_1year",
            hover_name="fund_symbol" if "fund_symbol" in etf.columns else None,
            title="Expense Ratio vs 1-Year Return",
            labels={"fund_annual_report_net_expense_ratio": "Expense Ratio",
                    "fund_return_1year": "1-Year Return (%)"},
            opacity=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)

    display_cols = [c for c in ["fund_symbol", "fund_short_name", "fund_category",
                                 "fund_return_1year", "fund_annual_report_net_expense_ratio",
                                 "total_net_assets"] if c in etf.columns]
    st.dataframe(etf[display_cols].head(50), use_container_width=True)


def render_complaint_dashboard(res):
    st.subheader("Complaint Dashboard")
    comp = res["comp"]

    product_dist = comp["Product"].value_counts().head(15)
    fig = px.bar(x=product_dist.values, y=product_dist.index, orientation="h",
                 title="Top 15 Complaint Categories",
                 labels={"x": "Count", "y": "Product"})
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    if "State" in comp.columns:
        state_dist = comp["State"].value_counts().head(10)
        fig2 = px.bar(x=state_dist.index, y=state_dist.values,
                      title="Top 10 States by Complaint Volume",
                      labels={"x": "State", "y": "Count"})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Classify a Complaint")
    complaint_text = st.text_area("Enter complaint text:")
    if complaint_text and st.button("Classify"):
        if "complaint_svm" in res:
            vec = res["comp_tfidf"].transform([clean_text(complaint_text)])
            pred = res["complaint_svm"].predict(vec)[0]
            label = res["comp_labels"].inverse_transform([pred])[0]
            st.success(f"Predicted Category: **{label}**")


def render_price_charts(res):
    st.subheader("Price Charts")
    prices = res["prices"]
    prices["Date"] = pd.to_datetime(prices["Date"])

    tickers = sorted(prices["ticker"].unique().tolist())
    selected_tickers = st.multiselect("Select Tickers", tickers,
                                       default=tickers[:5] if len(tickers) >= 5 else tickers)

    if selected_tickers:
        price_col = "Close" if "Close" in prices.columns else "Adj Close"
        subset = prices[prices["ticker"].isin(selected_tickers)]

        fig = px.line(subset, x="Date", y=price_col, color="ticker",
                      title="Price History")
        st.plotly_chart(fig, use_container_width=True)

        if "daily_return" in subset.columns:
            fig2 = px.line(subset, x="Date", y="daily_return", color="ticker",
                           title="Daily Returns")
            st.plotly_chart(fig2, use_container_width=True)


def render_sentiment_demo(res):
    st.subheader("Sentiment Analysis")
    text = st.text_input("Enter financial text to analyze:")
    if text:
        if "sentiment_svm" in res and "sentiment_tfidf" in res:
            vec = res["sentiment_tfidf"].transform([clean_text(text)])
            pred = res["sentiment_svm"].predict(vec)[0]
            label = res["sent_le"].inverse_transform([pred])[0]
            scores = res["sentiment_svm"].decision_function(vec)[0]

            color_map = {"positive": "green", "negative": "red", "neutral": "gray"}
            st.markdown(f"### Sentiment: :{color_map.get(label, 'blue')}[{label.upper()}]")

            score_dict = {res["sent_le"].inverse_transform([i])[0]: float(s)
                          for i, s in enumerate(scores)}
            fig = px.bar(x=list(score_dict.keys()), y=list(score_dict.values()),
                         title="Decision Function Scores",
                         labels={"x": "Sentiment", "y": "Score"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sentiment model not loaded. Run notebook 02_classification.ipynb first.")

    st.subheader("Recent Financial News Sentiment")
    news = res["news"]
    if "Sentiment" in news.columns:
        sent_dist = news["Sentiment"].value_counts()
        fig = px.pie(names=sent_dist.index, values=sent_dist.values,
                     title="News Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(news.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
