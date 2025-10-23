"""Preprocessing script for SMS spam dataset.
Reads input CSV, applies basic cleaning, and writes cleaned CSV.

Usage:
    python preprocessing.py --input D:\\test\\HW3\\sms_spam_no_header.csv --output D:\\test\\HW3\\sms_spam_clean.csv
"""
import argparse
import pandas as pd
import re


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.lower()
    # replace urls
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)
    # replace email addresses
    s = re.sub(r"\S+@\S+", " EMAIL ", s)
    # replace numbers
    s = re.sub(r"\d+", " NUM ", s)
    # remove non-word characters except spaces
    s = re.sub(r"[^\w\s]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_frame(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output cleaned CSV path")
    parser.add_argument("--text-col", default="text", help="Name of the text column")
    parser.add_argument("--label-col", default="label", help="Name of the label column")

    args = parser.parse_args()
    df = pd.read_csv(args.input, encoding="utf-8", header=0)

    # If there are only two columns unnamed, try to infer
    if args.text_col not in df.columns or args.label_col not in df.columns:
        # common format: label,text
        if df.shape[1] >= 2:
            df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})

    df_clean = preprocess_frame(df, text_col="text")
    df_clean.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote cleaned data to: {args.output}")


if __name__ == "__main__":
    main()
