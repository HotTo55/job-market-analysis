#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ====== files ======
INPUT_FILE = "/Users/totoh/Desktop/Unstructured Data/linkedin_raw.csv"
OUT_OVERALL = "/Users/totoh/Desktop/Unstructured Data/linkedin_tfidf_overall_skills.csv"
OUT_GROUP = "/Users/totoh/Desktop/Unstructured Data/linkedin_tfidf_skills_by_search_title.csv"

# ====== columns ======
TEXT_COL = "description"
GROUP_COL = "search_title"

# ====== skill dictionary ======
SKILL_TERMS = {
    "sql", "python", "r", "tableau", "power bi", "powerbi", "excel",
    "statistics", "machine learning", "ml", "ab testing", "a b testing",
    "etl", "airflow", "dbt", "databricks", "bigquery", "snowflake",
    "aws", "azure", "gcp", "looker", "power query", "sas", "spark",
    "pandas", "numpy", "dashboard", "dashboards", "kpi", "kpis",
    "forecasting", "experimentation", "data modeling", "data models",
    "reporting", "bi", "business intelligence", "api", "pipelines",
    "data pipeline", "data pipelines", "governance", "metadata",
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keep_skill(term):
    term = term.lower().strip()
    if term in SKILL_TERMS:
        return True
    for s in SKILL_TERMS:
        if " " in s and s in term:
            return True
    return False


df = pd.read_csv(INPUT_FILE)
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
cleaned = df[TEXT_COL].map(clean_text)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85,
    max_features=6000,
    sublinear_tf=True,
)

X = vectorizer.fit_transform(cleaned)
terms = np.array(vectorizer.get_feature_names_out())

# Overall skill TF-IDF
mean_scores = np.asarray(X.mean(axis=0)).ravel()
overall = pd.DataFrame({"term": terms, "mean_tfidf": mean_scores})
overall = overall[overall["term"].map(keep_skill)].sort_values("mean_tfidf", ascending=False)

# Skill TF-IDF by group (search_title)
rows = []
for group_name, group_idx in df.groupby(GROUP_COL).groups.items():
    subx = X[list(group_idx)]
    group_mean = np.asarray(subx.mean(axis=0)).ravel()
    temp = pd.DataFrame({"term": terms, "mean_tfidf": group_mean})
    temp = temp[temp["term"].map(keep_skill)]
    temp = temp[temp["mean_tfidf"] > 0].sort_values("mean_tfidf", ascending=False)
    temp[GROUP_COL] = group_name
    rows.append(temp[[GROUP_COL, "term", "mean_tfidf"]])

by_group = pd.concat(rows, ignore_index=True)

overall.to_csv(OUT_OVERALL, index=False)
by_group.to_csv(OUT_GROUP, index=False)

print("Done.")
print("Rows:", X.shape[0], "Features:", X.shape[1])
print("Saved overall:", OUT_OVERALL)
print("Saved by-group:", OUT_GROUP)
print("\nTop 20 overall skills:")
print(overall.head(20).to_string(index=False))
