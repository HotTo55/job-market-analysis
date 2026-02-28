#!/usr/bin/env python3
"""
Simple VADER sentiment analysis for reddit_raw.csv.

Outputs:
- reddit_sentiment_scored.csv
- reddit_sentiment_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
    parser = argparse.ArgumentParser(description="VADER sentiment on reddit CSV")
    parser.add_argument("--input", default="reddit_raw.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--out-scored", default="reddit_sentiment_scored.csv")
    parser.add_argument("--out-summary", default="reddit_sentiment_summary.csv")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Missing text column '{args.text_col}'. Available: {list(df.columns)}")

    analyzer = SentimentIntensityAnalyzer()
    scores = df[args.text_col].fillna("").astype(str).apply(analyzer.polarity_scores).apply(pd.Series)
    scored = pd.concat([df, scores], axis=1)
    scored["sentiment_label"] = pd.cut(
        scored["compound"],
        bins=[-1.0001, -0.05, 0.05, 1.0001],
        labels=["negative", "neutral", "positive"],
    )

    summary = build_summary(scored)

    scored.to_csv(args.out_scored, index=False)
    summary.to_csv(args.out_summary, index=False)

    overall = summary[summary["group"] == "overall"][["sentiment", "count", "share"]]
    print("Done.")
    print(f"Scored rows: {len(scored)}")
    print(f"Saved scored file: {args.out_scored}")
    print(f"Saved summary file: {args.out_summary}")
    print("\nOverall sentiment:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
