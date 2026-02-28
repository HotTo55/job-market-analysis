#!/usr/bin/env python3
"""
Run transformer-based sentiment analysis on reddit_raw.csv.

Outputs:
1) reddit_sentiment_transformer_scored.csv
2) reddit_sentiment_transformer_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _normalize_label(label: str) -> str:
    """Map model label variants to negative/neutral/positive."""
    l = label.strip().lower()
    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }
    return mapping.get(l, l)


def batched(items: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_sentiment(
    df: pd.DataFrame,
    text_col: str,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install first:\n"
            "  python3 -m pip install --user transformers torch tqdm"
        ) from exc

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):  # type: ignore
            return x

    texts = df[text_col].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id2label = model.config.id2label
    label_list = [_normalize_label(id2label[i]) for i in sorted(id2label)]

    preds = []
    probs = []

    for batch in tqdm(
        batched(texts, batch_size),
        total=int(np.ceil(len(texts) / batch_size)),
        desc="Scoring",
    ):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(p)
        preds.extend(p.argmax(axis=1).tolist())

    prob_arr = np.vstack(probs)
    pred_labels = [_normalize_label(id2label[i]) for i in preds]

    out = df.copy()
    out["sentiment_label"] = pred_labels

    for idx, lab in enumerate(label_list):
        out[f"prob_{lab}"] = prob_arr[:, idx]

    return out


def build_summary(scored: pd.DataFrame) -> pd.DataFrame:
    overall = (
        scored["sentiment_label"]
        .value_counts(dropna=False)
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    overall["share"] = overall["count"] / len(scored)
    overall["group"] = "overall"
    overall["group_value"] = "all"

    by_sub = (
        scored.groupby(["subreddit", "sentiment_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    by_sub["share"] = by_sub["count"] / by_sub.groupby("subreddit")["count"].transform("sum")
    by_sub = by_sub.rename(columns={"subreddit": "group_value", "sentiment_label": "sentiment"})
    by_sub["group"] = "subreddit"

    by_kw = (
        scored.groupby(["keyword", "sentiment_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    by_kw["share"] = by_kw["count"] / by_kw.groupby("keyword")["count"].transform("sum")
    by_kw = by_kw.rename(columns={"keyword": "group_value", "sentiment_label": "sentiment"})
    by_kw["group"] = "keyword"

    return pd.concat(
        [
            overall[["group", "group_value", "sentiment", "count", "share"]],
            by_sub[["group", "group_value", "sentiment", "count", "share"]],
            by_kw[["group", "group_value", "sentiment", "count", "share"]],
        ],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer sentiment on Reddit CSV")
    parser.add_argument(
        "--input",
        default="/Users/totoh/Desktop/Unstructured Data/reddit_raw.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Text column name",
    )
    parser.add_argument(
        "--model",
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max length (truncate longer posts)",
    )
    parser.add_argument(
        "--out-scored",
        default="/Users/totoh/Desktop/Unstructured Data/reddit_sentiment_transformer_scored.csv",
        help="Output path for scored rows",
    )
    parser.add_argument(
        "--out-summary",
        default="/Users/totoh/Desktop/Unstructured Data/reddit_sentiment_transformer_summary.csv",
        help="Output path for grouped summary",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Text column '{args.text_col}' not found. Available: {list(df.columns)}")

    scored = run_sentiment(
        df=df,
        text_col=args.text_col,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    summary = build_summary(scored)

    scored.to_csv(args.out_scored, index=False)
    summary.to_csv(args.out_summary, index=False)

    overall = summary[(summary["group"] == "overall")].copy()
    print("Done.")
    print(f"Scored rows: {len(scored)}")
    print(f"Saved scored file: {args.out_scored}")
    print(f"Saved summary file: {args.out_summary}")
    print("\nOverall sentiment:")
    print(overall[["sentiment", "count", "share"]].to_string(index=False))


if __name__ == "__main__":
    main()
