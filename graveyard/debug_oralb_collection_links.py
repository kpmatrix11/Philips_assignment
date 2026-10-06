from playwright.sync_api import sync_playwright

url = 'https://oralb.com/en-us/products/electric-toothbrushes/electric-toothbrushes/'
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1600, 'height': 1200}, locale='en-US')
    page.goto(url, wait_until='domcontentloaded', timeout=60000)
    page.wait_for_timeout(2500)
    links = page.locator('a[href*="/products/"]')
    print('total anchors', links.count())
    for i in range(min(50, links.count())):
        a = links.nth(i)
        href = a.get_attribute('href') or ''
        text = (a.inner_text() or '').replace('\n', ' | ')
        if '/products/electric-toothbrushes/' in href or '/products/electric-toothbrushes' in href:
            print('---', i, '---')
            print('href=', href)
            print('text=', text[:500])
    browser.close()
