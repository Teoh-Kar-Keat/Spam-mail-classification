"""Streamlit demo app for the spam classifier.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score)


sns.set(style="whitegrid")
st.set_page_config(layout="wide")

st.sidebar.title("HW3 Spam Classifier Demo")
csv_path = st.sidebar.text_input("Dataset CSV", value="sms_spam_clean.csv")
models_dir = st.sidebar.text_input("Models dir", value="models")
text_size = st.sidebar.number_input("Max text length (chars)", value=1000)
seed = st.sidebar.number_input("Seed", value=42)
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)
# Candidate columns suggested by user
preferred_text_cols = ["col_1", "col_0", "text_clean", "text_lower", "text_contracts_masked", "text_number", "text_stripped", "text_whitespace", "text_stopwords_removed"]
preferred_label_cols = ["col_1", "col_0", "label"]

st.sidebar.markdown("---")
st.sidebar.markdown("### Column selectors")

@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


df = load_data(csv_path)

def sidebar_column_selector(df, key_prefix=""):
    # create choices from dataframe columns plus preferred lists
    if df is not None:
        cols = list(dict.fromkeys(list(df.columns) + preferred_text_cols + preferred_label_cols))
    else:
        cols = preferred_text_cols + preferred_label_cols
    label_choice = st.sidebar.selectbox("Label column", cols, index=cols.index('label') if 'label' in cols else 0, key=key_prefix+"_label")
    text_choice = st.sidebar.selectbox("Text column", cols, index=cols.index('text_clean') if 'text_clean' in cols else (cols.index('text') if 'text' in cols else 1), key=key_prefix+"_text")
    # Do not call experimental_rerun directly inside helper — set a session flag instead
    if st.sidebar.button('Reload dataset'):
        st.session_state['reload_requested'] = True
    return label_choice, text_choice

label_col, text_col = sidebar_column_selector(df, key_prefix="main")

# If reload was requested, clear caches and rerun from main flow
if st.session_state.get('reload_requested'):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # reset flag then rerun
    st.session_state.pop('reload_requested', None)
    # reload dataset into df without calling experimental_rerun (safer for hosted runtimes)
    df = load_data(csv_path)

st.title("HW3 — Spam classifier demo")
st.markdown(f"**Using label column:** `{label_col}`  —  **text column:** `{text_col}`")
if df is None:
    st.info('Dataset not loaded. Enter a valid Dataset CSV path in the sidebar and press Reload dataset.')

model_path = os.path.join(models_dir, "logreg_pipeline.joblib")
model = load_model(model_path)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Controls & Status")
    if df is None:
        st.error("Could not load dataset. Check the Dataset CSV path in the sidebar.")
    else:
        st.write(f"Loaded {len(df)} rows from `{csv_path}`")
    if model is None:
        st.warning(f"Model not found at `{model_path}`. Train and save a model first.")
    else:
        st.success(f"Loaded model from `{model_path}`")

with col2:
    st.header("Data preview")
    if df is not None:
        st.dataframe(df.head())

st.markdown("---")

def top_tokens(series, n=30):
    c = Counter()
    for t in series.fillna('').astype(str):
        for w in t.split():
            c[w] += 1
    return c.most_common(n)

if df is not None:
    st.header("Data overview")
    # class distribution
    if label_col not in df.columns:
        st.warning(f"Selected label column `{label_col}` not found in dataset. Showing first column instead.")
        label_to_use = df.columns[0]
    else:
        label_to_use = label_col
    vc = df[label_to_use].astype(str).value_counts()
    st.subheader("Class distribution")
    st.bar_chart(vc)

    st.subheader("Token replacement approximate counts (top 20)")
    # approximate token counts
    if text_col not in df.columns:
        st.warning(f"Selected text column `{text_col}` not found in dataset. Showing first text-like column instead.")
        # try to guess a text column
        candidates = [c for c in df.columns if df[c].dtype == object]
        text_to_use = candidates[0] if candidates else df.columns[0]
    else:
        text_to_use = text_col
    top_all = top_tokens(df[text_to_use], n=20)
    ta = pd.DataFrame(top_all, columns=['token', 'count'])
    st.table(ta)

    # Token replacement counts (common masks)
    st.subheader('Token replacement counts (approx)')
    masks = ['URL', 'EMAIL', 'NUM']
    mask_counts = {m: 0 for m in masks}
    for tok, cnt in top_all:
        if tok in mask_counts:
            mask_counts[tok] = cnt
    st.table(pd.DataFrame(list(mask_counts.items()), columns=['token','count']))

    st.subheader("Top tokens by class (approx)")
    spam_texts = df[df[label_col].astype(str).str.lower().str.contains('spam')][text_col]
    ham_texts = df[~df[label_col].astype(str).str.lower().str.contains('spam')][text_col]
    top_spam = dict(top_tokens(spam_texts, n=30))
    top_ham = dict(top_tokens(ham_texts, n=30))
    tokens = list(set(list(top_spam.keys()) + list(top_ham.keys())))
    data = []
    for t in tokens:
        data.append({'token': t, 'spam': top_spam.get(t, 0), 'ham': top_ham.get(t, 0)})
    tf = pd.DataFrame(data).sort_values('spam', ascending=False).head(30)
    fig, ax = plt.subplots()
    tf.set_index('token')[['spam', 'ham']].plot(kind='bar', ax=ax)
    ax.set_title('Top tokens by class (approx)')
    st.pyplot(fig)

    st.markdown('---')

    if model is not None:
        st.header('Model performance on a held-out subset')
        # create a quick train/test split to evaluate
        X = df[text_col].astype(str)
        y = df[label_col].astype(str)
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = X, X, y, y

        preds = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            if 'spam' in model.classes_:
                prob_spam = model.predict_proba(X_test)[:, list(model.classes_).index('spam')]
            else:
                prob_spam = model.predict_proba(X_test)[:, 1]
        else:
            prob_spam = np.zeros(len(preds))

        st.subheader('Classification report')
        st.text(classification_report(y_test, preds))

        st.subheader('Confusion matrix')
        labels = list(model.classes_)
        cm = confusion_matrix(y_test, preds, labels=labels)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        st.subheader('ROC and Precision-Recall')
        fpr, tpr, _ = roc_curve((y_test == 'spam').astype(int), prob_spam)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve((y_test == 'spam').astype(int), prob_spam)
        ap = average_precision_score((y_test == 'spam').astype(int), prob_spam)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f'AP = {ap:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        st.pyplot(fig)

        st.subheader('Threshold sweep (precision/recall/f1)')
        thresholds = np.linspace(0, 1, 101)
        rows = []
        for t in thresholds:
            preds_t = (prob_spam >= t).astype(int)
            precision_v = precision_score((y_test == 'spam').astype(int), preds_t, zero_division=0)
            recall_v = recall_score((y_test == 'spam').astype(int), preds_t, zero_division=0)
            f1_v = f1_score((y_test == 'spam').astype(int), preds_t, zero_division=0)
            rows.append({'threshold': t, 'precision': precision_v, 'recall': recall_v, 'f1': f1_v})
        ts = pd.DataFrame(rows)
        fig, ax = plt.subplots()
        ax.plot(ts['threshold'], ts['precision'], label='precision')
        ax.plot(ts['threshold'], ts['recall'], label='recall')
        ax.plot(ts['threshold'], ts['f1'], label='f1')
        ax.set_xlabel('threshold')
        ax.set_ylabel('score')
        ax.legend()
        st.pyplot(fig)

st.markdown('---')

st.header('Live inference')
example_spam, example_ham = st.columns(2)
with example_spam:
    if st.button('Use spam example'):
        if df is not None:
            sp = df[df[label_col].astype(str).str.lower().str.contains('spam')]
            if not sp.empty:
                sample_text = sp.sample(1, random_state=seed).iloc[0][text_col]
                st.session_state['input_text'] = sample_text
with example_ham:
    if st.button('Use ham example'):
        if df is not None:
            ha = df[~df[label_col].astype(str).str.lower().str.contains('spam')]
            if not ha.empty:
                sample_text = ha.sample(1, random_state=seed).iloc[0][text_col]
                st.session_state['input_text'] = sample_text

text_input = st.text_area('Message to classify', value=st.session_state.get('input_text', ''), height=150)

if st.button('Predict'):
    if model is None:
        st.error('Model not found. Train and save a model to the models dir first.')
    else:
        proba = model.predict_proba([text_input])[0]
        labels = list(model.classes_)
        prob_spam_val = proba[labels.index('spam')] if 'spam' in labels else proba[1]
        st.write({labels[i]: float(proba[i]) for i in range(len(labels))})
        # show probability bar
        fig, ax = plt.subplots()
        ax.bar(labels, proba)
        ax.set_ylabel('Probability')
        st.pyplot(fig)

        # show spam probability over thresholds (mini chart)
        fig, ax = plt.subplots()
        ax.plot([0, threshold, 1], [0, prob_spam_val, 1], marker='o')
        ax.set_xlabel('Decision space (example)')
        ax.set_ylabel('Spam probability')
        ax.set_title(f'Spam probability = {prob_spam_val:.3f} (threshold {threshold:.2f})')
        st.pyplot(fig)

