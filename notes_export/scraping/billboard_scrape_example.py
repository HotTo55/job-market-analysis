import requests
from bs4 import BeautifulSoup
import pandas as pd

link = "https://www.billboard.com/charts/hot-100/"

link_request = requests.get(link)
soup = BeautifulSoup(link_request.content, 'html.parser')

artist_link = soup.select("a[href*='/artist/']")
artist_links = [x.get('href') for x in artist_link if "chart-history" not in x.get('href')]

songs = soup.select('.o-chart-results-list__item h3#title-of-a-story')

songs = [x.text.strip() for x in songs]

artists = soup.select('h3#title-of-a-story+.c-label')
artists = [x.text.strip() for x in artists]

pd.DataFrame({
    "Song": songs,
    "Artist": artists
})

entry_info = soup.select('.chart-results-container .o-chart-results-list-row-container .o-chart-results-list__item')

df_out = []

for row, song in enumerate(entry_info):
    link = song.select_one("a[href*='/artist/']")
    title = song.select_one('h3#title-of-a-story')
    artist = song.select_one('h3#title-of-a-story+.c-label')
    out = pd.DataFrame({
        "title": [title.text.strip() if title else None],
        "Artist": [artist.text.strip() if artist else None], 
        "link": [link.get('href') if link else None],
        'rank': [row + 1]
    })
    df_out.append(out)

df_out = pd.concat(df_out, ignore_index=True)

df_out.dropna(subset=['title'], inplace=True)

df_out.reset_index(drop=True, inplace=True)
df_out
df_out['rank'] = df_out.index + 1