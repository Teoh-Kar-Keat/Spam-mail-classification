"""Train a logistic regression model on cleaned SMS spam data.

Usage:
    python train.py --input D:\\test\\HW3\\sms_spam_clean.csv --model-dir models --text-col text --label-col label
"""
import argparse
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X = df[args.text_col].astype(str)
    y = df[args.label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "logreg_pipeline.joblib")
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
