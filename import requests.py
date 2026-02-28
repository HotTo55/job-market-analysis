import requests
from bs4 import BeautifulSoup
import pandas as pd

billboard_link = "https://www.billboard.com/charts/hot-100/"
billboard_request = requests.get(billboard_link)

billboard_soup = BeautifulSoup(billboard_request.content)

billboard_soup.select(".c-label")
song_titles = billboard_soup.select(".o-chart-results-list_item .c-label")

billboard_soup.select("#.o-chart-results-list_item #title-of-a-story")

songs[0].text.strip()

songs = [x.text.strip() for x in song_titles]

billboard_soup.select("#title-of-a-story+ .c-label")

[x.text.strip() for x in artists]

artist_links= billboard_soup.select("a[href*='/artist/']")

artist_links= [x.get('href') for x in artist_links]

pd.DataFrame({"artist:[artists],"
              "song":[songs]})

entry_soup= billboard_soup.select(".chart-results-list .o-chart-results-list-row-container .o-chart-results-list_item")
len(entry_soup)

df=[out]

for row, song in enumerate(entry_soup):
    title= song.select_one("#title-of-a-story")
    artist= song.select_one("#title-of-a-story.c-label")
    link= song.select_one("a[href*='/artist/']")
    out= pd.DataFrame({
        "title":[title.text.strip() if title else None],
        "artist":[artist.text.strip() if artist else None],
        "link:"[link.get('href') if link else None],
        "rank":[row + 1]
    })
    df.out = append(out)

df_out=pd.concat(df_out)

df_out.dropna(subset.['title'], inplace=True)

df_out.reset_index(drop=True, inplace=True)

df_out['rank'] = df_out.index + 1
