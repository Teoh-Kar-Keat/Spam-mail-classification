"""Streamlit demo app for the spam classifier.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(layout="wide")

st.sidebar.title("HW3 Spam Classifier Demo")
csv_path = st.sidebar.text_input("Dataset CSV", value="D:/test/HW3/sms_spam_clean.csv")
label_col = st.sidebar.text_input("Label column", value="label")
text_col = st.sidebar.text_input("Text column", value="text")
models_dir = st.sidebar.text_input("Models dir", value="models")
text_size = st.sidebar.number_input("Max text length (chars)", value=1000)
seed = st.sidebar.number_input("Seed", value=42)
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)


def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


df = load_data(csv_path)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Controls")
    if df is None:
        st.warning("Could not load dataset. Check path.")
    else:
        st.write(f"Loaded {len(df)} rows")

    model_path = os.path.join(models_dir, "logreg_pipeline.joblib")
    st.write(f"Model path: {model_path}")

with col2:
    st.header("Data overview")
    if df is not None:
        st.write(df.head())


st.header("Live Inference")
example_spam = st.button("Use spam example")
example_ham = st.button("Use ham example")
text_input = st.text_area("Message to classify", height=150)

if example_spam and df is not None:
    # try to pick a spam row
    sp = df[df[label_col].astype(str).str.lower().str.contains("spam")]
    if not sp.empty:
        text_input = sp.iloc[0][text_col]

if example_ham and df is not None:
    ha = df[~df[label_col].astype(str).str.lower().str.contains("spam")]
    if not ha.empty:
        text_input = ha.iloc[0][text_col]

if st.button("Predict"):
    if not os.path.exists(model_path):
        st.error("Model not found. Train and save a model to the models dir first.")
    else:
        model = joblib.load(model_path)
        prob = model.predict_proba([text_input])[0]
        labels = model.classes_
        st.write(dict(zip(labels, prob)))
