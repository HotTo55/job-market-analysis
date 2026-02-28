import pandas as pd
# The following gets ran in the terminal
# pip install playwright
# python3 -m playwright install
from playwright.sync_api import sync_playwright, Playwright
import re

# The following lines will activate a playwright
# browser, open a new page, and then go to the
# desired page.

pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False)

page = chrome.new_page()

page.goto('https://www.imdb.com/title/tt0290988/reviews/?ref_=tt_ov_ql_2')

# page.locator('css=.ipc-see-more__text').click()

# Display all reviews

# Watch for button name changes...
page.get_by_test_id("tturv-pagination").get_by_role("button", name="See All").click()

# The class below will grab the entire review block
# and the class can change on a whim. You'l have to
# inspect the page to find the correct class.

reviews = page.locator('css=.ipc-list-card__content')

# You'l always count your objects, for the sake
# of iterating over them.

reviews_count = reviews.count()

# You'll make constant use of the nth() function!

reviews.nth(0).hover()
reviews.nth(3).click()
reviews.nth(0).locator('css=.ipc-rating-star--rating').inner_text()

# The is_visible function is very useful when you have to
# include something condition (i.e., if no star is visible,
# then return None in your DataFrame).

reviews.locator('css=.ipc-list-card--border-speech').nth(3).locator('css=.ipc-rating-star').is_visible()

# Lots of flexibility with how you find elements!
# Most of the time you are looking for the inner text.

review = reviews.nth(0).get_by_test_id('review-overflow').inner_text()

# You'll always have to create a list to your data!

review_list = []

for i in range(0, reviews_count):

    # A fun surprise! Some reviews have a spoiler button!
    if reviews.nth(i).get_by_role("button", name="Spoiler").is_visible():
        reviews.nth(i).get_by_role("button", name="Spoiler").click()
        review_text = reviews.nth(i).get_by_test_id('review-spoiler-content').inner_text()
    else:
        # This pulls from all none spoiler reviews
        review_text = reviews.nth(i).get_by_test_id('review-overflow').inner_text()
    #review_text = reviews.nth(i).locator('css=.ipc-list-card__content').inner_text()
    # Hitting some cleanup!
    review_text = re.sub(r'\n', ' ', review_text)
    # You'll usually have to create a DataFrame for each review
    review_df = pd.DataFrame({'review': [review_text]})
    review_list.append(review_df)
    
pd.concat(review_list)

page.get_by_placeholder('Search IMDb').fill('Barbie')

page.locator('css=#suggestion-search-button').click()
page.get_by_test_id('find-results-section-title').get_by_text('Barbie').nth(0).click()

page.locator('css=.ipc-link').count()

page.close()

chrome.close()

pw.stop()
