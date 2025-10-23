# HW3 — SMS Spam Classifier (Logistic Regression)

This repository contains an OpenSpec-driven homework project to build a spam/ham classifier using logistic regression.

Contents:
- `preprocessing.py` — CSV cleaning pipeline
- `train.py` — trains a TF-IDF + logistic regression pipeline and saves the model
- `predict.py` — predict probabilities for a single message or a CSV
- `app.py` — Streamlit demo app skeleton
- `requirements.txt` — Python dependencies
- `sms_spam_no_header.csv` — (your dataset, placed at root)

Phases:
1. Preprocessing
2. Train & predict
3. Notebook with visualization (in Traditional Chinese)
4. Deployment (Streamlit app)
5. Report (how OpenSpec was used)

Run locally (example):
```powershell
python preprocessing.py --input D:\\test\\HW3\\sms_spam_no_header.csv --output D:\\test\\HW3\\sms_spam_clean.csv
python train.py --input D:\\test\\HW3\\sms_spam_clean.csv --model-dir models
streamlit run app.py
```
