#!/usr/bin/env python3
"""Scrape Billboard Hot 100 (robust example).

This script is defensive about selectors and saves results to CSV.
Adjust CSS selectors in `song_selectors` / `artist_selectors` if Billboard's
HTML structure changes.
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.billboard.com/charts/hot-100/"


def fetch(url: str) -> bytes:
    resp = requests.get(url, timeout=10, headers={"User-Agent": "python-requests/2.x"})
    resp.raise_for_status()
    return resp.content


def extract(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    # A few candidate selectors; keep them adjustable if site changes.
    song_selectors = [
        "h3#title-of-a-story",
        "h3.c-title",
        ".o-chart-results-list__item h3",
        "li.o-chart-results-list__item h3",
    ]

    artist_selectors = [
        ".o-chart-results-list__item span.c-label",
        ".o-chart-results-list__item .c-label",
        ".o-chart-results-list__item .c-label.a-no-trucate",
    ]

    def find_first_nonempty(selectors: list[str]) -> list[BeautifulSoup]:
        for sel in selectors:
            elems = soup.select(sel)
            if elems:
                return elems
        return []

    song_elems = find_first_nonempty(song_selectors)
    artist_elems = find_first_nonempty(artist_selectors)

    songs = [e.get_text(strip=True) for e in song_elems]
    artists = [e.get_text(strip=True) for e in artist_elems]

    # Fallback: iterate row containers and extract title/artist pairs
    if not songs or not artists:
        rows = soup.select(".o-chart-results-list-row-container")
        for row in rows:
            s = row.select_one("h3")
            a = row.select_one(".c-label")
            if s and a:
                songs.append(s.get_text(strip=True))
                artists.append(a.get_text(strip=True))

    n = min(len(songs), len(artists))
    return artists[:n], songs[:n]


def main() -> None:
    html = fetch(URL)
    soup = BeautifulSoup(html, "html.parser")
    artists, songs = extract(soup)

    if not songs:
        print("No songs found. Check selectors or network access.")
        return

    df = pd.DataFrame({"artist": artists, "song": songs})
    print(df.head())
    df.to_csv("billboard_hot100.csv", index=False)
    print(f"Saved billboard_hot100.csv with {len(df)} rows")


if __name__ == "__main__":
    main()
