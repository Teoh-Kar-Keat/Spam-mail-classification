"""Predict script for the trained logistic regression pipeline.

Usage:
    python predict.py --model models/logreg_pipeline.joblib --text "Free money!!!"
    python predict.py --model models/logreg_pipeline.joblib --input D:\\test\\HW3\\sms_spam_clean.csv --text-col text
"""
import argparse
import joblib
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", help="Single text to predict")
    parser.add_argument("--input", help="CSV file with texts to predict")
    parser.add_argument("--text-col", default="text")
    args = parser.parse_args()

    model = joblib.load(args.model)

    if args.text:
        prob = model.predict_proba([args.text])[0]
        labels = model.classes_
        print(dict(zip(labels, prob)))
        return

    if args.input:
        df = pd.read_csv(args.input)
        texts = df[args.text_col].astype(str).tolist()
        probs = model.predict_proba(texts)
        out = pd.DataFrame(probs, columns=[f"prob_{c}" for c in model.classes_])
        out = pd.concat([df.reset_index(drop=True), out], axis=1)
        out_path = os.path.splitext(args.input)[0] + "_preds.csv"
        out.to_csv(out_path, index=False)
        print(f"Wrote predictions to: {out_path}")
        return

    print("Provide --text or --input to predict")


if __name__ == "__main__":
    main()
