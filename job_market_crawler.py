#!/usr/bin/env python3
"""
Simple job market crawler:
1) Reddit posts via PRAW
2) Indeed search result snippets via requests + BeautifulSoup

Outputs default to:
- reddit_raw.csv
- indeed_raw.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


@dataclass
class RedditConfig:
    subreddits: list[str]
    keywords: list[str]
    start_date: dt.datetime
    end_date: dt.datetime
    limit_per_query: int


def _to_utc_epoch(d: dt.datetime) -> int:
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp())


def crawl_reddit(cfg: RedditConfig) -> pd.DataFrame:
    try:
        import praw
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for Reddit crawling.\n"
            "Install: python3 -m pip install --user praw"
        ) from exc

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "job-market-analysis-script")
    if not client_id or not client_secret:
        raise SystemExit(
            "Set env vars REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET first."
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    start_epoch = _to_utc_epoch(cfg.start_date)
    end_epoch = _to_utc_epoch(cfg.end_date)
    rows: list[dict] = []

    for sub in cfg.subreddits:
        sr = reddit.subreddit(sub)
        for kw in cfg.keywords:
            # PRAW's search "new" order is simple and stable for this use case.
            query = kw
            got = 0
            for post in sr.search(query, sort="new", time_filter="all", limit=cfg.limit_per_query):
                created = int(post.created_utc)
                if created < start_epoch or created > end_epoch:
                    continue
                text = f"{post.title or ''} {post.selftext or ''}".strip()
                rows.append(
                    {
                        "subreddit": sub,
                        "keyword": kw,
                        "post_id": post.id,
                        "title": post.title,
                        "selftext": post.selftext,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": dt.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S"),
                        "text": text,
                        "year_month": dt.datetime.utcfromtimestamp(created).strftime("%Y-%m"),
                        "url": f"https://www.reddit.com{post.permalink}",
                    }
                )
                got += 1
            print(f"[reddit] r/{sub} | '{kw}' -> {got} posts in range")
            time.sleep(0.8)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    return df


def _safe_text(node) -> str:
    if not node:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def crawl_indeed(
    queries: Iterable[str],
    location: str,
    pages_per_query: int,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    rows: list[dict] = []
    for q in queries:
        for page in range(pages_per_query):
            start = page * 10
            url = "https://www.indeed.com/jobs"
            params = {"q": q, "l": location, "start": start}
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            cards = soup.select("div.job_seen_beacon, div.cardOutline, div.result")
            for card in cards:
                title_node = card.select_one("h2.jobTitle span[title], h2.jobTitle a span, h2.jobTitle")
                company_node = card.select_one("span.companyName, .companyName")
                loc_node = card.select_one("div.companyLocation, .companyLocation")
                snippet_node = card.select_one("div.job-snippet, .job-snippet")
                date_node = card.select_one("span.date, .date")
                salary_node = card.select_one("div.salary-snippet, .salary-snippet-container, .salary-snippet")
                link_node = card.select_one("h2.jobTitle a, a.jcs-JobTitle")

                link = ""
                if link_node and link_node.has_attr("href"):
                    href = link_node["href"]
                    link = f"https://www.indeed.com{href}" if href.startswith("/") else href

                rows.append(
                    {
                        "query": q,
                        "title": _safe_text(title_node),
                        "company": _safe_text(company_node),
                        "location": _safe_text(loc_node),
                        "salary": _safe_text(salary_node),
                        "date_posted": _safe_text(date_node),
                        "snippet": _safe_text(snippet_node),
                        "job_url": link,
                        "crawl_date": dt.datetime.utcnow().strftime("%Y-%m-%d"),
                    }
                )
            print(f"[indeed] '{q}' page {page + 1}/{pages_per_query} -> {len(cards)} cards")
            time.sleep(sleep_seconds)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["query", "title", "company", "location", "job_url"]).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Reddit + Indeed crawler")
    parser.add_argument("--reddit-out", default="reddit_raw.csv")
    parser.add_argument("--indeed-out", default="indeed_raw.csv")
    parser.add_argument(
        "--subreddits",
        default="cscareerquestions,jobs,MBA",
        help="Comma-separated subreddit names",
    )
    parser.add_argument(
        "--keywords",
        default="job search,laid off,no response,rejected,offer,unemployment,ghosted,hiring freeze",
        help="Comma-separated search keywords",
    )
    parser.add_argument("--start-date", default="2023-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-12-31", help="YYYY-MM-DD")
    parser.add_argument("--reddit-limit", type=int, default=800, help="Max posts per subreddit+keyword query")
    parser.add_argument("--indeed-queries", default="business analyst,data analyst")
    parser.add_argument("--indeed-location", default="United States")
    parser.add_argument("--indeed-pages", type=int, default=10, help="Pages per Indeed query")
    parser.add_argument("--skip-reddit", action="store_true")
    parser.add_argument("--skip-indeed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_reddit:
        cfg = RedditConfig(
            subreddits=[x.strip() for x in args.subreddits.split(",") if x.strip()],
            keywords=[x.strip() for x in args.keywords.split(",") if x.strip()],
            start_date=dt.datetime.strptime(args.start_date, "%Y-%m-%d"),
            end_date=dt.datetime.strptime(args.end_date, "%Y-%m-%d"),
            limit_per_query=args.reddit_limit,
        )
        rdf = crawl_reddit(cfg)
        rdf.to_csv(args.reddit_out, index=False)
        print(f"Saved Reddit rows: {len(rdf)} -> {args.reddit_out}")

    if not args.skip_indeed:
        idf = crawl_indeed(
            queries=[x.strip() for x in args.indeed_queries.split(",") if x.strip()],
            location=args.indeed_location,
            pages_per_query=args.indeed_pages,
        )
        idf.to_csv(args.indeed_out, index=False)
        print(f"Saved Indeed rows: {len(idf)} -> {args.indeed_out}")


if __name__ == "__main__":
    main()
