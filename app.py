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

DEFAULT_CSV = "sms_spam_clean.csv"
DEFAULT_MODELS = "models"

# Layout: main content (left) and controls (right)
main_col, control_col = st.columns([3, 1])

with control_col:
    st.header("Controls")
    csv_path = st.text_input("Dataset CSV", value=st.session_state.get('csv_path', DEFAULT_CSV))
    models_dir = st.text_input("Models dir", value=st.session_state.get('models_dir', DEFAULT_MODELS))
    text_size = st.number_input("Max text length (chars)", value=st.session_state.get('text_size', 1000))
    seed = st.number_input("Seed", value=st.session_state.get('seed', 42))
    threshold = st.slider("Decision threshold", 0.0, 1.0, float(st.session_state.get('threshold', 0.5)))
    st.markdown("---")
    # persistent storage
    st.session_state['csv_path'] = csv_path
    st.session_state['models_dir'] = models_dir
    st.session_state['text_size'] = text_size
    st.session_state['seed'] = seed
    st.session_state['threshold'] = threshold

    # Reload / refresh buttons
    if st.button('Reload dataset'):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()
    if st.button('Reload model'):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()


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


# load dataset and model after reading controls
df = load_data(csv_path)
model_path = os.path.join(models_dir, "logreg_pipeline.joblib")
model = load_model(model_path)

with main_col:
    st.title("HW3 — Spam classifier demo")
    st.markdown("### Column selectors (visible)")
    if df is not None:
        cols = df.columns.tolist()
        label_col = st.selectbox("Label column", cols, index=cols.index('label') if 'label' in cols else 0)
        text_col = st.selectbox("Text column", cols, index=cols.index('text') if 'text' in cols else min(1, len(cols)-1))
    else:
        st.info('Dataset not loaded. Enter a valid Dataset CSV path at the right and press Reload dataset.')
        label_col = st.text_input("Label column", value='label')
        text_col = st.text_input("Text column", value='text')

    col_preview, col_empty = st.columns([2, 1])
    with col_preview:
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
    vc = df[label_col].astype(str).value_counts()
    st.subheader("Class distribution")
    st.bar_chart(vc)

    st.subheader("Token replacement approximate counts (top 20)")
    # approximate token counts
    top_all = top_tokens(df[text_col], n=20)
    ta = pd.DataFrame(top_all, columns=['token', 'count'])
    st.table(ta)

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

