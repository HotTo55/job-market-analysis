###################################
### Getting Box Scores and      ###
### Play by Play Data for       ###
### Notre Dame WBB from ESPN    ###
###################################


import asyncio
from playwright.async_api import async_playwright, Playwright
from bs4 import BeautifulSoup
import pandas as pd
import re
from io import StringIO

## We need to get game links first ##

wbb_box_link = 'https://www.espn.com/womens-college-basketball/team/schedule/_/id/87'

async def get_page_title(url):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            page_html = await page.locator("body").inner_html()
            wbb_soup = BeautifulSoup(page_html, 'html.parser')
            game_ids = wbb_soup.select("a[href*='gameId']")
            await p.stop()
            await page.close()
            await browser.close()
            return game_ids

if __name__ == "__main__":
    url_to_test = wbb_box_link
    game_ids = asyncio.run(get_page_title(url_to_test))

id_links = [x.get('href') for x in game_ids]

id_links

id_links = [re.sub(r'\bgame\b', 'boxscore', x) for x in id_links]
id_links = [re.sub(r'(?<=\d/).*$', '', x) for x in id_links]

## Then we can get box scores ##

async def get_boxscores(url):
        wbb_stats = pd.read_html(url, match="MIN")
        wbb_players = pd.read_html(url, match="starters")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto(url)
            table_rows = page.locator(".BoxscoreItem__TeamName")
            place = page.locator(".NzyJW.fuwnA")
            print(await place.count())
            rows = [] 
            for i in range(await table_rows.count()):
                row = await table_rows.nth(i).inner_text()
                place_text = await place.nth(i*3).inner_text()
                print(place_text)
                box_score = pd.concat([wbb_players[i], wbb_stats[i]], axis=1, ignore_index=True)
                box_score[0] = box_score[0].str.replace(r'[A-Z]\. .*\d+$', '', regex=True)
                box_score.columns = box_score.loc[0]
                box_score = box_score[~box_score['starters'].isin(['starters', 'bench', 'team'])]
                box_score['team'] = row
                box_score['location'] = place_text
                box_score['game_url'] = url
                rows.append(box_score)
            tables = pd.concat(rows)
            await p.stop()
            await page.close()
            await browser.close()
            return tables

if __name__ == "__main__":
    url_to_test = id_links[0]
    box_scores = asyncio.run(get_boxscores(url_to_test))

empty_games = []

for link in id_links:
    try:
        box_scores = asyncio.run(get_boxscores(link))
        empty_games.append(box_scores)
    except Exception as e:
        print(f"Error processing {link}: {e}")

all_box_scores = pd.concat(empty_games, ignore_index=True)       

all_box_scores
all_box_scores.to_csv('/Users/sberry5/Documents/wbb_box_scores.csv', index=False)

## And finally, we can get play by play data ##

async def get_pbp(url):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto(url)
            period_rows = page.locator(".tabs__list__item")
            period_count = await period_rows.count()
            rows = [] 
            for i in range(period_count):
                await period_rows.nth(i).click()
                row_count = await page.locator(".playByPlay__tableRow").count()
                plays = []
                logos = []
                for k in range(row_count):
                    if await page.locator(".Image.Logo.Logo__sm").nth(k).is_visible():
                        logo = await page.locator(".Image.Logo.Logo__sm").nth(k).get_attribute("src")
                    else:
                        logo = None
                    play = await page.locator(".playByPlay__tableRow").nth(k).inner_text()
                    plays.append(play)
                    logos.append(logo)
                plays_df = pd.DataFrame({'play': plays, 'logo': logos})
                plays_df['period'] = i + 1
                rows.append(plays_df)
            pbp_table = pd.concat(rows, ignore_index=True)
            pbp_table['game_url'] = url
            await p.stop()
            await page.close()
            await browser.close()
            return pbp_table

if __name__ == "__main__":
    url_to_test = "https://www.espn.com/womens-college-basketball/playbyplay/_/gameId/401812420"
    pbp = asyncio.run(get_pbp(url_to_test))

pbp['logo'] = pbp['logo'].str.extract(r'(\d+)(?=\.png)')
pbp['is_nd'] = pbp['logo'].apply(lambda x: 1 if x == "87" else 0)

pbp['to_ignore'] = pbp['play'].str.contains('subbing')

pbp_links = [re.sub(r'\bboxscore\b', 'playbyplay', x) for x in id_links]    

pbp_games = []

for link in pbp_links:
    try:
        pbp_table = asyncio.run(get_pbp(link))
        pbp_games.append(pbp_table)
    except Exception as e:
        print(f"Error processing {link}: {e}")

pbp_all = pd.concat(pbp_games, ignore_index=True)
pbp_all['logo'] = pbp_all['logo'].str.extract(r'(\d+)(?=\.png)')        
pbp_all['is_nd'] = pbp_all['logo'].apply(lambda x: 1 if x == "87" else 0)

pbp_all.to_csv('/Users/sberry5/Documents/wbb_pbp.csv', index=False)

####################
### Getting Pace ###
####################

import requests
from bs4 import BeautifulSoup
import numpy as np
import time
import pandas as pd

years = range(2020, 2026)

adv_list = []

for year in years:
    link = f"https://www.sports-reference.com/cbb/schools/notre-dame/women/{year}-gamelogs-advanced.html"
    adv_stats = pd.read_html(link, header=1)[0]
    adv_stats['year'] = year
    adv_list.append(adv_stats)
    time.sleep(np.random.uniform(1,3))


all_adv_stats = pd.concat(adv_list, ignore_index=True)

all_adv_stats.to_csv('/Users/sberry5/Documents/nd_wbb_advanced_stats.csv', index=False)