import sys
sys.path.append('src')
from step_1_2_oralb_scrapper import extract_listing_price, clean_text
from playwright.sync_api import sync_playwright
import re

url = 'https://oralb.com/en-us/products/electric-toothbrushes/electric-toothbrushes/'
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1600, 'height': 1200}, locale='en-US')
    page.goto(url, wait_until='domcontentloaded', timeout=60000)
    page.wait_for_timeout(2000)
    body = clean_text(page.locator('body').inner_text()) or ''
    title = 'Oral-B iO Series 10 Twin Pack, Cosmic Black + Cosmic Black'
    print('body_has_title', title in body)
    idx = body.find(title)
    print('title_index', idx)
    window = body[idx: idx + 600] if idx != -1 else body[:600]
    print(window)
    print('regex_match', re.search(rf'{re.escape(title)}.*?\$\s*(\d[\d,]*(?:\.\d{{1,2}})?)', body, flags=re.I | re.S))
    print('extract_listing_price', extract_listing_price(page, title, 'https://oralb.com/en-us/products/oral-b-io-series-10-twin-pack-cosmic-black-cosmic-black'))
    browser.close()
