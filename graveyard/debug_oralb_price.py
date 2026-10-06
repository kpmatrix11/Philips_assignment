import re
from playwright.sync_api import sync_playwright

url = 'https://oralb.com/en-us/products/electric-toothbrushes/electric-toothbrushes/'
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1600, 'height': 1200}, locale='en-US')
    page.goto(url, wait_until='domcontentloaded', timeout=60000)
    page.wait_for_timeout(2500)

    body_text = page.locator('body').inner_text()
    print('body_has_series10', 'Series 10' in body_text)
    print('price_tokens', re.findall(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', body_text)[:20])
    for term in ['Series 10', 'Series 9', 'Series 8', 'Series 7', 'Series 6', 'Series 5']:
        idx = body_text.find(term)
        print('\nTERM', term, 'IDX', idx)
        if idx != -1:
            print(body_text[max(0, idx - 300): idx + 1200])

    links = page.locator('a[href*="/products/"]')
    print('anchor_count', links.count())
    for i in range(min(12, links.count())):
        link = links.nth(i)
        text = (link.inner_text() or '').replace('\n', ' | ')
        href = link.get_attribute('href')
        if 'series' in (text + ' ' + (href or '')).lower() or 'io' in (text + ' ' + (href or '')).lower():
            print('---', i + 1, '---')
            print('link_text=', text[:500])
            print('href=', href)
            parent = link.locator('xpath=ancestor::*[self::article or self::li or self::div or self::section][1]')
            if parent.count():
                print('parent_text=', (parent.inner_text() or '')[:1500])
    browser.close()
