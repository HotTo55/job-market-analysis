from playwright.sync_api import sync_playwright, Playwright
import re

pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False)

page = chrome.new_page()

page.goto("https://www.espn.com/nfl/teams")

link_count = page.locator("css=a[href*='roster']").count()

team_links = []*link_count

for i in range(link_count):
    link = page.locator("css=a[href*='roster']").nth(i).get_attribute('href')
    link = 'https://www.espn.com' + link
    team_links.append(link)

player_links = []

for i in range(len(team_links)):    
    page.goto(team_links[i])    

    player_locator = page.locator("css=.inline.Table__TD--headshot a[href*='player/_/id']")

    player_count = player_locator.count()

    for j in range(player_count):
        link = player_locator.nth(j).get_attribute('href')
        link = re.sub(r'(.*player/)(_/.*)', '\\1bio/\\2', link)
        player_links.append(link)

page.goto("https://www.espn.com/nfl/player/bio/_/id/4240414/zach-morton")

page.get_by_text(re.compile("\w+, [A-Z]{2}")).inner_text()

key_content = page.locator('h1') 

key = key_content.inner_text()

page_content = page.locator('.LQUZJ')

text_contents = page_content.all_inner_texts()

output = []
for i in range(1, page_content.__len__()):
    row_info = text_contents[i]
    row_info = re.sub(r',', '', row_info)
    row_info = re.sub(r'\n', ',', row_info)        
    row_df = pd.read_csv(StringIO(row_info), header=None)
    if row_df.shape[1] == 4:
        row_df.columns = ['ARTIST', 'SONG', 'RATING', 'HITS']
    else:
        row_df.columns = ['ARTIST', 'SONG', 'HITS']
        row_df['RATING'] = np.nan
    output.append(row_df)


chrome.close()

pw.stop()

pd.concat(output)