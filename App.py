import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import openai  
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import transformers

# ✅ Download the VADER lexicon (needed for sentiment analysis)
nltk.download('vader_lexicon')

# ✅ Streamlit page setup
st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")

# ✅ Fetch API Key from Streamlit secrets (ensure it's set in your secrets.toml file)
API_KEY = st.secrets.get("api_keys", {}).get("GROQ_API_KEY", None)

if not API_KEY:
    st.error("⚠ Groq API Key not found! Please check Streamlit secrets.")
    st.stop()

# ✅ Fetch Slack Webhook from Streamlit secrets (ensure it's set in your secrets.toml file)
SLACK_WEBHOOK = st.secrets.get("api_keys", {}).get("SLACK_WEBHOOK", None)

if not SLACK_WEBHOOK:
    st.error("⚠️ Slack Webhook URL is missing. Please check your configuration.")
    st.stop()

# ✅ Function to generate response using Groq API
def generate_response(user_input):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": user_input}],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_json = response.json()
        
        if "choices" in response_json and response_json["choices"]:
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"Error: Groq API response was empty. Response: {response_json}"
    except requests.exceptions.RequestException as e:
        return f"Error: Request to Groq API failed: {e}"

# ✅ Streamlit User Interface
st.title("Competitor Strategy Tracker")
user_query = st.text_input("Enter your query:")

# ✅ Button to trigger AI response
if st.button("Get Insights"):
    if not user_query.strip():
        st.warning("⚠ Please enter a valid query!")  
    else:
        result = generate_response(user_query)
        st.text_area("AI Response", result, height=200)

# ✅ Function to send data to Slack
def send_to_slack(data):
    """Send generated data to a Slack channel with error handling."""
    if not SLACK_WEBHOOK:
        st.error("⚠️ Slack Webhook URL is missing. Please check your configuration.")
        return

    payload = {"text": data}
    
    try:
        response = requests.post(
            SLACK_WEBHOOK,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            st.success("✅ Recommendations sent to Slack successfully!")
        else:
            st.error(f"⚠️ Slack API Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network error while sending to Slack: {e}")

# ✅ Sentiment Analysis using VADER
def analyze_sentiment(reviews):
    if not reviews:  # If reviews is empty, return a message
        return "No reviews available"
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for review in reviews:
        score = analyzer.polarity_scores(review)
        if score['compound'] >= 0.05:
            sentiments.append("Positive")
        elif score['compound'] <= -0.05:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

# ✅ Load and preprocess data
def load_and_preprocess_data(file_path, drop_na_columns=None):
    """Load and preprocess data from a CSV file."""
    data = pd.read_csv(file_path)
    if drop_na_columns:
        data = data.dropna(subset=drop_na_columns)
    return data

# ✅ Function to train predictive model
def train_predictive_model(data):
    """Train a predictive model to estimate competitor discount strategies."""
    
    if data.empty:
        st.error("⚠ Error: Competitor data is empty. Cannot train model.")
        return None, None

    data["discount"] = pd.to_numeric(data["discount"].str.replace("%", "", regex=True), errors="coerce").fillna(0)
    data["price"] = pd.to_numeric(data["price"], errors="coerce").fillna(0).astype(int)

    required_cols = ["price", "discount"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"⚠ Error: Missing required columns: {missing_cols}")
        return None, None

    X = data[required_cols]
    y = data["discount"] + (data["price"] * 0.05).round(2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    data["Predicted_Discount"] = model.predict(X)
    return model, data

# ✅ Load competitor data
competitor_data = load_and_preprocess_data("competitor_data.csv", drop_na_columns=["date", "discount"])

if competitor_data.empty:
    st.error("⚠️ Competitor data is empty. Please check the file.")
else:
    products = competitor_data["title"].dropna().unique().tolist()

# ✅ Display product selection
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

if selected_product:
    competitor_data_filtered = competitor_data[competitor_data["title"] == selected_product]
    st.subheader(f"Competitor Analysis for {selected_product}")
    st.write(competitor_data_filtered.tail(5))

# ✅ Perform sentiment analysis
product_reviews = []
if selected_product:
    reviews_data = load_and_preprocess_data("reviews.csv")
    product_reviews = reviews_data[reviews_data["title"] == selected_product]["review_statements"].dropna().tolist()

if product_reviews:
    sentiments = analyze_sentiment(product_reviews)
    sentiment_df = pd.DataFrame(sentiments, columns=["Sentiment"])
    st.write("Sentiment Analysis Results:", sentiment_df)
else:
    st.warning("⚠️ No reviews found for sentiment analysis.")

# ✅ Visualize sentiment results
if not sentiment_df.empty:
    fig = px.bar(sentiment_df['Sentiment'].value_counts().reset_index(), x="index", y="Sentiment", title="Sentiment Analysis")
    st.plotly_chart(fig)

# ✅ Forecasting Competitor Discounts using ARIMA
def forecast_discounts_arima(data, future_days=5):
    if data.empty:
        st.error("⚠️ Competitor data is empty. Cannot forecast.")
        return pd.DataFrame(columns=["Date", "Predicted_Discount"]).set_index("Date")

    data["discount"] = pd.to_numeric(data["discount"], errors="coerce")
    discount_series = data["discount"].dropna()

    if len(discount_series) < 5:
        st.warning("⚠️ Not enough data for forecasting.")
        return pd.DataFrame(columns=["Date", "Predicted_Discount"]).set_index("Date")

    model = ARIMA(discount_series, order=(2, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days)

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})
    forecast_df.set_index("Date", inplace=True)

    return forecast_df

# ✅ Show ARIMA forecasting if there's enough data
if competitor_data_filtered.empty:
    st.warning("⚠️ No competitor data available for forecasting.")
else:
    competitor_data_with_predictions = forecast_discounts_arima(competitor_data_filtered)
    st.write("Forecasted Discounts:", competitor_data_with_predictions.tail(10))

# ✅ Send recommendations to Slack
if sentiment_df.empty:
    st.warning("⚠️ No sentiment data available to generate recommendations.")
else:
    recommendations = generate_response(f"Provide strategic recommendations based on competitor data and sentiment analysis.")
    send_to_slack(recommendations)
    st.write("Recommendations Sent to Slack!")
