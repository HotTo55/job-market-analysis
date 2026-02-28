import requests
import pandas as pd
import time
from datetime import datetime

headers    = {"User-Agent": "Mozilla/5.0 job_market_study"}
subreddits = ["cscareerquestions","jobs","h1b","internationalstudents","datascience",
              "businessanalysis"]
keywords= ["H1B","visa","sponsorship","OPT","data analyst","business analyst","data science",
            "machine learning","entry level","new grad","STEM extension"]
records = []

for sub in subreddits:
    for kw in subreddits:
        print(f"Scraping r/{sub} | '{kw}' ...")
        
        after = None
        
        for page in range(5):
            params = {"q": kw, "sort": "new", "t": "year", 
                      "limit": 100, "restrict_sr": True}
            
            if after:
                params["after"] = after
            
            try:
                url = f"https://www.reddit.com/r/{sub}/search.json"
                r   = requests.get(url, headers=headers, params=params, timeout=10)
                data = r.json()
            except:
                break
            
            posts = data["data"]["children"]
            if not posts:
                break
            
            for post in posts:
                p = post["data"]
                records.append({
                    "subreddit"    : sub,
                    "keyword"      : kw,
                    "post_id"      : p.get("id"),
                    "title"        : p.get("title", ""),
                    "selftext"     : p.get("selftext", ""),
                    "score"        : p.get("score", 0),
                    "num_comments" : p.get("num_comments", 0),
                    "created_utc"  : datetime.utcfromtimestamp(p.get("created_utc", 0)),
                })
            
            after = data["data"].get("after")
            if not after:
                break
            
            time.sleep(2)

#  Clean up 
df = pd.DataFrame(records).drop_duplicates(subset="post_id")
df["text"]       = df["title"] + " " + df["selftext"].fillna("")
df["year_month"] = df["created_utc"].dt.to_period("M")

df.to_csv("/Users/totoh/Desktop/Unstructured Data/reddit_raw.csv", index=False)
print(f"\n✓ Total posts: {len(df):,}")
print(df.groupby("subreddit")["post_id"].count())


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

df["sentiment"] = df["text"].apply(
    lambda x: vader.polarity_scores(x)["compound"]
)

sentiment_trend = df.groupby("year_month")["sentiment"].mean()

sentiment_trend.plot(figsize=(10,5))