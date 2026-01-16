import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import re
import time

# --- 1. SETTINGS & MODEL LOADING (Requirement 1.iii) ---
st.set_page_config(page_title="NLP Dashboard JIE43303", layout="wide")

@st.cache_resource
def load_nlp_brain():
    # Loading two models to show "Advanced Features"
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    return sentiment_model, emotion_model

sent_pipe, emot_pipe = load_nlp_brain()

# --- 2. PREPROCESSING FUNCTION (Technical Quality Rubric) ---
def clean_text(text):
    text = re.sub(r'<br />', ' ', text) # Remove IMDb HTML tags
    text = re.sub(r'[^a-zA-Z ]', '', text) # Remove special characters
    return text.lower().strip()

# --- 3. DASHBOARD UI ---
st.title("📊 NLP Sentiment & Emotion Dashboard")
st.markdown("### Course Project: JIE43303 Natural Language Processing")

# Sidebar
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload IMDb CSV", type="csv")

# Tabs
tab_individual, tab_batch = st.tabs(["Single Review Tester", "Batch Analysis & Performance"])

# FEATURE 1: Individual Analysis
with tab_individual:
    user_input = st.text_area("Paste review text here:", "The acting was amazing but the story was a bit slow.")
    if st.button("Analyze Now"):
        cleaned = clean_text(user_input)
        s_res = sent_pipe(cleaned)[0]
        e_res = emot_pipe(cleaned)[0]
        
        c1, c2 = st.columns(2)
        c1.metric("Sentiment Polarity", s_res['label'], f"{s_res['score']:.2%}")
        c2.metric("Emotion Detected", e_res['label'], f"{e_res['score']:.2%}")

# FEATURE 2: Batch Analysis & MEASUREMENT (Requirement 1.iv)
with tab_batch:
    if uploaded_file:
        df = pd.read_csv(uploaded_file).head(30) # Processing 30 rows for demo speed
        if st.button("Run Model Measurement"):
            with st.spinner("Processing..."):
                start = time.time()
                df['prediction'] = df['review'].apply(lambda x: sent_pipe(clean_text(x[:512]))[0]['label'].lower())
                end = time.time()
                
                # Performance Math
                correct = (df['sentiment'] == df['prediction']).sum()
                accuracy = (correct / len(df)) * 100
                
                # Display Metrics
                m1, m2 = st.columns(2)
                m1.success(f"System Accuracy: {accuracy:.2f}%")
                m2.info(f"Speed: {len(df)/(end-start):.2f} reviews/sec")
                
                # Visuals
                fig = px.bar(df['prediction'].value_counts(), title="Sentiment Distribution")
                st.plotly_chart(fig)
                st.dataframe(df[['review', 'sentiment', 'prediction']])
    else:
        st.warning("Please upload the IMDB CSV file in the sidebar.")
