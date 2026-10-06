from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import pandas as pd
from playwright.sync_api import (
    BrowserContext,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


# ============================================================
# CONFIGURATION
# ============================================================

PHILIPS_DOMAIN = "https://www.usa.philips.com"

LISTING_BASE_URL = (
    "https://www.usa.philips.com/"
    "c-m-pe/electric-toothbrushes/latest"
)

TOTAL_LISTING_PAGES = None
MAX_LISTING_PAGES = 50
EXPECTED_PRODUCTS = 29

HEADLESS = True

OUTPUT_DIR = Path("data/philips")

PAGE_TIMEOUT_MS = 60_000

WAIT_AFTER_NAVIGATION_MS = 2_000

WAIT_AFTER_CLICK_MS = 500


# ============================================================
# PRODUCT MODEL
# ============================================================

@dataclass
class PhilipsProduct:

    brand: str = "Philips Sonicare"

    # ---------------------------------------------
    # Basic listing information
    # ---------------------------------------------

    product_name: Optional[str] = None
    product_type: Optional[str] = None

    model_codes: list[str] = field(
        default_factory=list
    )

    price_usd: Optional[float] = None

    listing_features: list[str] = field(
        default_factory=list
    )

    listing_attributes: dict = field(
        default_factory=dict
    )

    image_urls: list[str] = field(
        default_factory=list
    )

    listing_url: Optional[str] = None
    product_url: Optional[str] = None

    raw_listing_text: Optional[str] = None

    # ---------------------------------------------
    # Product detail information
    # ---------------------------------------------

    detail_title: Optional[str] = None
    product_category: Optional[str] = None
    breadcrumb_path: list[str] = field(
        default_factory=list
    )
    availability_status: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    product_features: list[str] = field(
        default_factory=list
    )
    product_colors: list[str] = field(
        default_factory=list
    )
    rating_value: Optional[float] = None
    rating_count: Optional[int] = None
    product_variants: list[dict] = field(
        default_factory=list
    )

    json_ld: list[dict] = field(
        default_factory=list
    )

    detail_sections: dict = field(
        default_factory=dict
    )

    technical_specifications: dict = field(
        default_factory=dict
    )

    detail_links: list[dict] = field(
        default_factory=list
    )

    detail_images: list[str] = field(
        default_factory=list
    )

    raw_detail_text: Optional[str] = None
    detail_metadata: dict = field(
        default_factory=dict
    )

    # ---------------------------------------------
    # Metadata
    # ---------------------------------------------

    scraped_at: Optional[str] = None

    scrape_errors: list[str] = field(
        default_factory=list
    )


# ============================================================
# BASIC HELPERS
# ============================================================

def now_utc() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(
    value: Optional[str],
) -> Optional[str]:

    if value is None:
        return None

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def unique_strings(
    values: list[str],
) -> list[str]:

    result = []
    seen = set()

    for value in values:

        value = clean_text(value)

        if not value:
            continue

        key = value.lower()

        if key not in seen:

            seen.add(key)
            result.append(value)

    return result


def parse_price(
    text: Optional[str],
) -> Optional[float]:

    if not text:
        return None

    match = re.search(
        r"\$\s*([\d,]+(?:\.\d{1,2})?)",
        text,
    )

    if not match:
        return None

    return float(
        match.group(1).replace(",", "")
    )


def extract_model_codes(
    text: Optional[str],
) -> list[str]:

    if not text:
        return []

    matches = re.findall(
        r"\b(?:HX|HY)[A-Z0-9_]+(?:/[A-Z0-9_]+)?\b",
        text,
        flags=re.I,
    )

    return unique_strings(
        [
            value.upper()
            for value in matches
        ]
    )


def absolute_url(
    url: Optional[str],
) -> Optional[str]:

    if not url:
        return None

    return urljoin(
        PHILIPS_DOMAIN,
        url,
    )


def build_listing_url(
    page_number: int,
) -> str:

    return (
        f"{LISTING_BASE_URL}"
        f"?availability=instock"
        f"&page={page_number}"
    )


def extract_page_number_from_url(
    url: Optional[str],
) -> Optional[int]:

    if not url:
        return None

    match = re.search(
        r"[?&]page=(\d+)",
        url,
        re.I,
    )

    if not match:
        return None

    return int(match.group(1))


def find_next_listing_page(
    page: Page,
    current_page: int,
) -> Optional[int]:

    selectors = [
        "a[href*='page=']",
        "a[href*='latest']",
        "a[aria-label*='next' i]",
        "a[aria-label*='page' i]",
        "button[aria-label*='next' i]",
    ]

    best_page = None

    for selector in selectors:

        links = page.locator(selector)

        for index in range(links.count()):

            link = links.nth(index)

            try:
                href = link.get_attribute("href")
            except Exception:
                continue

            if not href:
                continue

            next_page = extract_page_number_from_url(
                absolute_url(href)
            )

            if next_page is None:
                continue

            if next_page > current_page:
                if best_page is None or next_page < best_page:
                    best_page = next_page

    return best_page


# ============================================================
# LISTING ATTRIBUTE NORMALIZATION
# ============================================================

def derive_listing_attributes(
    features: list[str],
) -> dict:
    """
    Convenience normalization only.

    The original feature strings remain authoritative.

    IMPORTANT:
    Missing feature != False.
    """

    attributes = {}

    for feature in features:

        lower = feature.lower()

        # Brush modes

        match = re.search(
            r"(\d+)\s+"
            r"(?:brush\s+)?modes?",
            lower,
        )

        if match:

            attributes[
                "brush_modes_count"
            ] = int(match.group(1))

        # Intensities

        match = re.search(
            r"(\d+)\s+intensit(?:y|ies)",
            lower,
        )

        if match:

            attributes[
                "intensities_count"
            ] = int(match.group(1))

        # Battery

        match = re.search(
            r"(\d+)[-\s]*day"
            r".{0,30}battery",
            lower,
        )

        if match:

            attributes[
                "battery_life_days"
            ] = int(match.group(1))

        # Pressure

        if "pressure sensor" in lower:

            attributes[
                "pressure_sensor"
            ] = feature

        # Timers

        timers = []

        if "smartimer" in lower:
            timers.append("SmarTimer")

        if "quadpacer" in lower:
            timers.append("QuadPacer")

        if "timer" in lower and not timers:
            timers.append(feature)

        if timers:

            existing = attributes.get(
                "timers",
                [],
            )

            attributes["timers"] = (
                unique_strings(
                    existing + timers
                )
            )

        # App / Bluetooth

        if (
            "sonicare app" in lower
            or "bluetooth" in lower
            or "connected app" in lower
        ):

            attributes[
                "app_connectivity"
            ] = feature

        # Claims

        claim_types = {
            "plaque": "plaque_claims",
            "gum": "gum_claims",
            "white": "whitening_claims",
            "stain": "stain_removal_claims",
        }

        for keyword, key in (
            claim_types.items()
        ):

            if keyword in lower:

                attributes.setdefault(
                    key,
                    [],
                ).append(feature)

    return attributes


# ============================================================
# PAGE SCROLLING
# ============================================================

def scroll_page(
    page: Page,
    max_steps: int = 30,
) -> None:

    try:

        page.evaluate(
            "window.scrollTo(0, 0)"
        )

    except Exception:
        return

    page.wait_for_timeout(300)

    previous_height = -1

    for _ in range(max_steps):

        try:

            height = page.evaluate(
                "document.body.scrollHeight"
            )

            page.evaluate(
                """
                window.scrollBy(
                    0,
                    Math.max(
                        window.innerHeight * 0.8,
                        700
                    )
                )
                """
            )

            page.wait_for_timeout(250)

            position = page.evaluate(
                """
                window.scrollY +
                window.innerHeight
                """
            )

            if position >= height:

                page.wait_for_timeout(500)

                new_height = page.evaluate(
                    "document.body.scrollHeight"
                )

                if new_height == previous_height:
                    break

                previous_height = new_height

        except Exception:
            break


# ============================================================
# LISTING CARD DISCOVERY
# ============================================================

def find_product_cards(
    page: Page,
) -> list[Locator]:

    selectors = [
        "article",
        "[data-testid*='product']",
        "[class*='product-card']",
        "[class*='productCard']",
        "[class*='ProductCard']",
        "li",
    ]

    candidates = []

    seen_texts = set()

    for selector in selectors:

        locator = page.locator(selector)

        for index in range(
            locator.count()
        ):

            node = locator.nth(index)

            try:

                if not node.is_visible():
                    continue

                text = clean_text(
                    node.inner_text(
                        timeout=500
                    )
                )

            except Exception:
                continue

            if not text:
                continue

            # Product card signals.

            has_model = bool(
                re.search(
                    r"\b(?:HX|HY)[A-Z0-9]+"
                    r"(?:/[A-Z0-9]+)?\b",
                    text,
                    re.I,
                )
            )

            has_price = bool(
                re.search(
                    r"\$\s*\d+",
                    text,
                )
            )

            if not has_model:
                continue

            # Price is not always rendered in the
            # compact product card text on Philips.
            # The model code + "View product" CTA is still
            # a reliable product signal.

            # Avoid obvious huge parent containers.

            if len(text) > 2500:
                continue

            text_key = text.lower()

            if text_key in seen_texts:
                continue

            seen_texts.add(text_key)

            candidates.append(node)

    return candidates


# ============================================================
# LISTING NAME
# ============================================================

def extract_listing_name(
    card: Locator,
    lines: list[str],
) -> Optional[str]:

    headings = card.locator(
        "h1, h2, h3, h4, "
        "[role='heading']"
    )

    for i in range(
        headings.count()
    ):

        try:

            value = clean_text(
                headings.nth(i).inner_text(
                    timeout=300
                )
            )

        except Exception:
            continue

        if not value:
            continue

        lower = value.lower()

        if (
            "sonicare" in lower
            or "toothbrush" in lower
        ):

            return value

    for line in lines:

        if "sonicare" in line.lower():
            return line

    return None


# ============================================================
# LISTING PRODUCT TYPE
# ============================================================

def extract_listing_type(
    lines: list[str],
) -> Optional[str]:

    keywords = [
        "rechargeable toothbrush",
        "sonic electric toothbrush",
        "electric toothbrush",
        "power toothbrush",
    ]

    for line in lines:

        lower = line.lower()

        if any(
            keyword in lower
            for keyword in keywords
        ):

            return line

    return None


# ============================================================
# LISTING FEATURES
# ============================================================

def extract_listing_features(
    card: Locator,
) -> list[str]:

    features = []

    list_items = card.locator("li")

    for i in range(
        list_items.count()
    ):

        try:

            value = clean_text(
                list_items.nth(i).inner_text(
                    timeout=300
                )
            )

        except Exception:
            continue

        if (
            value
            and len(value) <= 350
        ):

            features.append(value)

    return unique_strings(features)


# ============================================================
# PRODUCT URL
# ============================================================

def extract_product_url(
    card: Locator,
) -> Optional[str]:

    links = card.locator(
        "a[href]"
    )

    fallback = []

    for i in range(
        links.count()
    ):

        link = links.nth(i)

        href = link.get_attribute(
            "href"
        )

        if not href:
            continue

        url = absolute_url(href)

        fallback.append(url)

        # Philips product detail URLs shown by the user.
        if "/c-p/" in href:

            return url

    # Secondary check.

    for url in fallback:

        if "/c-p/" in url:

            return url

    # Model-containing URL fallback.

    for url in fallback:

        if re.search(
            r"(?:HX|HY)[A-Z0-9]",
            url,
            re.I,
        ):

            return url

    return None


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_images(
    container: Locator,
) -> list[str]:

    images = []

    nodes = container.locator(
        "img"
    )

    for i in range(
        nodes.count()
    ):

        image = nodes.nth(i)

        for attribute in [
            "src",
            "data-src",
            "data-lazy-src",
        ]:

            value = image.get_attribute(
                attribute
            )

            if value:

                images.append(
                    absolute_url(value)
                )

        srcset = image.get_attribute(
            "srcset"
        )

        if srcset:

            for item in srcset.split(","):

                candidate = (
                    item.strip()
                    .split(" ")[0]
                )

                if candidate:

                    images.append(
                        absolute_url(
                            candidate
                        )
                    )

    return unique_strings(images)


# ============================================================
# LISTING CARD EXTRACTION
# ============================================================

def extract_listing_card(
    card: Locator,
    listing_url: str,
) -> Optional[PhilipsProduct]:

    try:

        raw_text = clean_text(
            card.inner_text(
                timeout=1000
            )
        )

    except Exception:
        return None

    if not raw_text:
        return None

    models = extract_model_codes(
        raw_text
    )

    if not models:
        return None

    lines = unique_strings(
        raw_text.splitlines()
    )

    features = (
        extract_listing_features(
            card
        )
    )

    return PhilipsProduct(
        product_name=extract_listing_name(
            card,
            lines,
        ),

        product_type=extract_listing_type(
            lines
        ),

        model_codes=models,

        price_usd=parse_price(
            raw_text
        ),

        listing_features=features,

        listing_attributes=(
            derive_listing_attributes(
                features
            )
        ),

        image_urls=extract_images(
            card
        ),

        listing_url=listing_url,

        product_url=extract_product_url(
            card
        ),

        raw_listing_text=raw_text,

        scraped_at=now_utc(),
    )


# ============================================================
# PRODUCT DEDUPLICATION
# ============================================================

def product_key(
    product: PhilipsProduct,
) -> Optional[str]:

    if product.product_url:
        return product.product_url

    if product.model_codes:
        return product.model_codes[0]

    if product.product_name:
        return product.product_name.lower()

    return None


def product_richness_score(
    product: PhilipsProduct,
) -> int:

    return (
        len(product.listing_features)
        + len(product.listing_attributes)
        + len(product.image_urls)
        + (5 if product.product_url else 0)
        + (3 if product.product_name else 0)
    )


def deduplicate_products(
    products: list[PhilipsProduct],
) -> list[PhilipsProduct]:

    output = {}

    for product in products:

        key = product_key(product)

        if not key:
            continue

        existing = output.get(key)

        if existing is None:

            output[key] = product
            continue

        if (
            product_richness_score(product)
            >
            product_richness_score(existing)
        ):

            output[key] = product

    return list(
        output.values()
    )


# ============================================================
# LISTING SCRAPER
# ============================================================

def scrape_listing_page(
    page: Page,
    page_number: int,
) -> list[PhilipsProduct]:

    url = build_listing_url(
        page_number
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"LISTING PAGE {page_number}"
    )

    print(url)

    page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=PAGE_TIMEOUT_MS,
    )

    page.wait_for_timeout(
        WAIT_AFTER_NAVIGATION_MS
    )

    scroll_page(page)

    cards = find_product_cards(
        page
    )

    print(
        f"Candidate cards: "
        f"{len(cards)}"
    )

    products = []

    for index, card in enumerate(
        cards,
        start=1,
    ):

        try:

            product = (
                extract_listing_card(
                    card,
                    url,
                )
            )

            if not product:
                continue

            products.append(product)

            print(
                f"  {index:02d}. "
                f"{product.product_name} "
                f"{product.model_codes} "
                f"${product.price_usd}"
            )

        except Exception as exc:

            print(
                f"  Card {index} error: "
                f"{exc}"
            )

    products = (
        deduplicate_products(
            products
        )
    )

    print(
        f"Unique products: "
        f"{len(products)}"
    )

    return products


# ============================================================
# JSON-LD
# ============================================================

def extract_json_ld(
    page: Page,
) -> list[dict]:

    output = []

    scripts = page.locator(
        "script[type='application/ld+json']"
    )

    for i in range(
        scripts.count()
    ):

        try:

            raw = scripts.nth(i).text_content()

            if not raw:
                continue

            data = json.loads(raw)

            if isinstance(data, dict):
                output.append(data)

            elif isinstance(data, list):

                output.extend(
                    [
                        item
                        for item in data
                        if isinstance(item, dict)
                    ]
                )

        except Exception:
            continue

    return output


# ============================================================
# SHOW MORE
# ============================================================

def click_show_more_buttons(
    page: Page,
) -> None:

    selectors = [
        "button:has-text('Show more')",
        "button:has-text('Show More')",
        "[role='button']:has-text('Show more')",
        "[role='button']:has-text('Show More')",
    ]

    clicked = set()

    for _ in range(10):

        found_new = False

        for selector in selectors:

            buttons = page.locator(
                selector
            )

            for i in range(
                buttons.count()
            ):

                button = buttons.nth(i)

                try:

                    if not button.is_visible():
                        continue

                    text = clean_text(
                        button.inner_text()
                    )

                    key = (
                        text,
                        i,
                    )

                    if key in clicked:
                        continue

                    button.scroll_into_view_if_needed()

                    page.wait_for_timeout(
                        200
                    )

                    button.click(
                        timeout=3000
                    )

                    clicked.add(key)

                    found_new = True

                    page.wait_for_timeout(
                        WAIT_AFTER_CLICK_MS
                    )

                except Exception:
                    continue

        if not found_new:
            break


# ============================================================
# NORMAL DETAIL PAGE SECTIONS
# ============================================================

def extract_page_sections(
    page: Page,
) -> dict:
    """
    Preserve sections dynamically.

    We intentionally do not define expected heading names.
    """

    result = {}

    headings = page.locator(
        "h2, h3, h4"
    )

    for i in range(
        headings.count()
    ):

        heading = headings.nth(i)

        try:

            if not heading.is_visible():
                continue

            title = clean_text(
                heading.inner_text(
                    timeout=300
                )
            )

        except Exception:
            continue

        if not title:
            continue

        try:

            content = heading.evaluate(
                """
                element => {

                    const output = [];

                    let node =
                        element.nextElementSibling;

                    while (node) {

                        if (
                            ['H2', 'H3', 'H4']
                            .includes(node.tagName)
                        ) {
                            break;
                        }

                        const text =
                            node.innerText?.trim();

                        if (text) {
                            output.push(text);
                        }

                        node =
                            node.nextElementSibling;
                    }

                    return output;
                }
                """
            )

        except Exception:

            content = []

        cleaned = unique_strings(
            content
        )

        if not cleaned:
            continue

        # Handle duplicate heading names.

        key = title

        counter = 2

        while key in result:

            key = (
                f"{title} "
                f"({counter})"
            )

            counter += 1

        result[key] = cleaned

    return result


# ============================================================
# PRODUCT DETAIL METADATA
# ============================================================

def extract_meta_value(
    page: Page,
    selectors: list[str],
) -> Optional[str]:

    for selector in selectors:

        locator = page.locator(selector)

        for i in range(locator.count()):

            value = locator.nth(i).get_attribute(
                "content"
            )

            if value:
                return clean_text(value)

    return None


def extract_breadcrumbs(
    page: Page,
) -> list[str]:

    crumbs = []

    selectors = [
        "nav a",
        "[class*='breadcrumb'] a",
        "[aria-label*='breadcrumb' i] a",
        "[itemprop='itemListElement'] a",
    ]

    for selector in selectors:

        nodes = page.locator(selector)

        for i in range(nodes.count()):

            node = nodes.nth(i)

            try:
                text = clean_text(
                    node.inner_text(timeout=200)
                )
            except Exception:
                continue

            if text and text.lower() not in {"home", "products"}:
                crumbs.append(text)

    return unique_strings(crumbs)


def extract_aggregate_rating_from_json_ld(
    json_ld: list[dict],
) -> tuple[Optional[float], Optional[int]]:

    def find_rating_block(node):

        if isinstance(node, dict):

            if "aggregateRating" in node:
                return node["aggregateRating"]

            for value in node.values():
                result = find_rating_block(value)
                if result is not None:
                    return result

        elif isinstance(node, list):

            for value in node:
                result = find_rating_block(value)
                if result is not None:
                    return result

        return None

    aggregate = find_rating_block(json_ld)

    if not isinstance(aggregate, dict):
        return None, None

    rating_value = aggregate.get("ratingValue")
    review_count = aggregate.get("reviewCount")

    if review_count is None:
        review_count = aggregate.get("ratingCount")

    try:
        rating = float(rating_value) if rating_value is not None else None
    except (TypeError, ValueError):
        rating = None

    try:
        count = int(float(review_count)) if review_count is not None else None
    except (TypeError, ValueError):
        count = None

    return rating, count


def extract_rating_summary(
    page: Page,
    json_ld: Optional[list[dict]] = None,
) -> tuple[Optional[float], Optional[int]]:

    rating = None
    count = None

    if json_ld:
        json_rating, json_count = extract_aggregate_rating_from_json_ld(json_ld)
        if json_rating is not None:
            rating = json_rating
        if json_count is not None:
            count = json_count

    for selector, attr in [
        ("meta[itemprop='ratingValue']", "content"),
        ("meta[property='og:rating']", "content"),
        ("[itemprop='ratingValue']", "content"),
    ]:

        loc = page.locator(selector)
        for i in range(loc.count()):
            value = loc.nth(i).get_attribute(attr)
            if value:
                try:
                    rating = float(value)
                    break
                except Exception:
                    pass
        if rating is not None:
            break

    for selector, attr in [
        ("meta[itemprop='ratingCount']", "content"),
        ("meta[itemprop='reviewCount']", "content"),
        ("[itemprop='ratingCount']", "content"),
    ]:

        loc = page.locator(selector)
        for i in range(loc.count()):
            value = loc.nth(i).get_attribute(attr)
            if value:
                try:
                    count = int(float(value))
                    break
                except Exception:
                    pass
        if count is not None:
            break

    return rating, count


def extract_product_features(
    page: Page,
) -> list[str]:

    features = []

    locations = [
        page.locator("li"),
        page.locator("[class*='feature']"),
        page.locator("[class*='benefit']"),
        page.locator("[class*='spec']"),
    ]

    for loc in locations:

        for i in range(loc.count()):

            try:
                text = clean_text(loc.nth(i).inner_text(timeout=200))
            except Exception:
                continue

            if not text:
                continue

            if len(text) <= 250:
                features.append(text)

    return unique_strings(features)


def extract_json_ld_colors(
    json_ld: Optional[list[dict]],
) -> list[str]:

    colors: list[str] = []

    def walk(node):

        if isinstance(node, dict):
            for key in [
                "color",
                "colors",
                "colorOption",
                "colorOptions",
                "colorSwatch",
                "colorName",
            ]:
                value = node.get(key)
                if isinstance(value, str):
                    colors.extend(
                        [
                            item.strip()
                            for item in re.split(r"[|,/]+", value)
                            if item.strip()
                        ]
                    )
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            colors.extend(
                                [
                                    part.strip()
                                    for part in re.split(r"[|,/]+", item)
                                    if part.strip()
                                ]
                            )
                        elif isinstance(item, dict):
                            walk(item)
                elif isinstance(value, dict):
                    walk(value)

            for value in node.values():
                walk(value)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(json_ld)

    return unique_strings(colors)


def extract_product_colors(
    page: Page,
    json_ld: Optional[list[dict]] = None,
) -> list[str]:

    colors: list[str] = []

    if json_ld:
        colors.extend(
            extract_json_ld_colors(json_ld)
        )

    locators = [
        page.locator("[aria-label*='color' i]"),
        page.locator("[data-testid*='color' i]"),
        page.locator("[data-color]"),
        page.locator("[class*='color' i]"),
        page.locator("button[title]"),
        page.locator("button[aria-label]"),
        page.locator("[name*='color' i]"),
    ]

    for loc in locators:

        for i in range(loc.count()):

            try:
                text = clean_text(
                    loc.nth(i).get_attribute("aria-label")
                    or loc.nth(i).get_attribute("title")
                    or loc.nth(i).get_attribute("data-color")
                    or loc.nth(i).get_attribute("value")
                    or loc.nth(i).inner_text(timeout=200)
                )
            except Exception:
                continue

            if not text:
                continue

            parts = [
                part.strip()
                for part in re.split(r"[|,/]+", text)
                if part.strip()
            ]

            for part in parts:
                if any(
                    word in part.lower()
                    for word in [
                        "black",
                        "white",
                        "blue",
                        "silver",
                        "gold",
                        "pink",
                        "purple",
                        "gray",
                        "rose",
                        "red",
                        "green",
                        "brown",
                        "beige",
                        "ivory",
                        "navy",
                        "graphite",
                        "brown",
                        "champagne",
                        "charcoal",
                    ]
                ):
                    colors.append(part)

    return unique_strings(colors)


def extract_availability_status(
    page: Page,
) -> Optional[str]:

    text = clean_text(page.locator("body").inner_text(timeout=2000))

    if not text:
        return None

    lower = text.lower()

    if "out of stock" in lower:
        return "out_of_stock"

    if "in stock" in lower:
        return "in_stock"

    if "preorder" in lower:
        return "preorder"

    if "backorder" in lower:
        return "backorder"

    return None


def extract_detail_metadata(
    page: Page,
    json_ld: list[dict],
) -> dict:

    metadata = {}

    description = extract_meta_value(
        page,
        [
            "meta[name='description']",
            "meta[property='og:description']",
            "meta[name='twitter:description']",
        ],
    )

    rating_value, rating_count = extract_rating_summary(page, json_ld)

    if description:
        metadata["description"] = description

    metadata["breadcrumbs"] = extract_breadcrumbs(page)
    metadata["availability"] = extract_availability_status(page)
    metadata["rating_summary"] = {
        "rating_value": rating_value,
        "rating_count": rating_count,
    }

    for item in json_ld:

        if not isinstance(item, dict):
            continue

        for key in [
            "brand",
            "category",
            "sku",
            "gtin",
            "mpn",
            "productID",
            "availability",
            "price",
            "priceCurrency",
            "url",
        ]:

            if key in item and item[key] not in (None, ""):
                metadata.setdefault("json_ld", {})[key] = item[key]

    return metadata


def extract_detail_links(
    page: Page,
) -> list[dict]:

    output = []

    seen = set()

    links = page.locator(
        "a[href]"
    )

    for i in range(
        links.count()
    ):

        link = links.nth(i)

        try:

            href = link.get_attribute(
                "href"
            )

            if not href:
                continue

            url = absolute_url(
                href
            )

            text = clean_text(
                link.inner_text(
                    timeout=200
                )
            )

            key = (
                text,
                url,
            )

            if key in seen:
                continue

            seen.add(key)

            output.append(
                {
                    "text": text,
                    "url": url,
                }
            )

        except Exception:
            continue

    return output


# ============================================================
# OPEN TECHNICAL SPECIFICATIONS
# ============================================================

def open_technical_specifications(
    page: Page,
) -> bool:

    print(
        "    Opening technical specifications..."
    )

    selectors = [
        "button:has-text('Technical Specifications')",
        "button:has-text('Technical specifications')",
        "[role='button']:has-text('Technical Specifications')",
        "[role='button']:has-text('Technical specifications')",
    ]

    for selector in selectors:

        locator = page.locator(
            selector
        )

        for i in range(
            locator.count()
        ):

            button = locator.nth(i)

            try:

                if not button.is_visible():
                    continue

                button.scroll_into_view_if_needed()

                page.wait_for_timeout(
                    300
                )

                button.click(
                    timeout=5000
                )

                page.wait_for_timeout(
                    700
                )

                return True

            except Exception:
                continue

    return False


# ============================================================
# FIND TECHNICAL DRAWER
# ============================================================

def find_technical_dialog(
    page: Page,
) -> Optional[Locator]:

    selectors = [
        "[role='dialog']",
        "[aria-modal='true']",
        "[class*='drawer']",
        "[class*='Drawer']",
        "[class*='modal']",
        "[class*='Modal']",
    ]

    for selector in selectors:

        candidates = page.locator(
            selector
        )

        for i in range(
            candidates.count()
        ):

            candidate = (
                candidates.nth(i)
            )

            try:

                if not candidate.is_visible():
                    continue

                text = clean_text(
                    candidate.inner_text(
                        timeout=500
                    )
                )

                if not text:
                    continue

                if (
                    "technical specifications"
                    in text.lower()
                ):

                    return candidate

            except Exception:
                continue

    return None


# ============================================================
# SCROLL TECHNICAL DRAWER
# ============================================================

def scroll_dialog_to_bottom(
    dialog: Locator,
) -> None:

    try:

        dialog.evaluate(
            """
            element => {

                const nodes = [
                    element,
                    ...element.querySelectorAll('*')
                ];

                for (const node of nodes) {

                    if (
                        node.scrollHeight >
                        node.clientHeight + 50
                    ) {

                        node.scrollTop =
                            node.scrollHeight;
                    }
                }
            }
            """
        )

    except Exception:
        pass


def scroll_dialog_to_top(
    dialog: Locator,
) -> None:

    try:

        dialog.evaluate(
            """
            element => {

                const nodes = [
                    element,
                    ...element.querySelectorAll('*')
                ];

                for (const node of nodes) {

                    if (
                        node.scrollHeight >
                        node.clientHeight + 50
                    ) {

                        node.scrollTop = 0;
                    }
                }
            }
            """
        )

    except Exception:
        pass


# ============================================================
# EXPAND TECHNICAL GROUPS
# ============================================================

def expand_all_technical_groups(
    page: Page,
    dialog: Locator,
) -> None:

    # Multiple passes because expanding one section
    # can expose/re-render other sections.

    for _ in range(4):

        controls = dialog.locator(
            "[aria-expanded], "
            "button"
        )

        changed = False

        for i in range(
            controls.count()
        ):

            control = controls.nth(i)

            try:

                if not control.is_visible():
                    continue

                text = clean_text(
                    control.inner_text(
                        timeout=200
                    )
                )

                if not text:
                    continue

                lower = text.lower()

                if lower in {
                    "technical specifications",
                    "close",
                    "add to cart",
                    "where to buy",
                }:
                    continue

                expanded = (
                    control.get_attribute(
                        "aria-expanded"
                    )
                )

                if expanded == "false":

                    control.scroll_into_view_if_needed()

                    control.click(
                        timeout=2000
                    )

                    page.wait_for_timeout(
                        150
                    )

                    changed = True

            except Exception:
                continue

        if not changed:
            break


# ============================================================
# LABEL/VALUE EXTRACTION
# ============================================================

def extract_label_value_pairs(
    container: Locator,
) -> dict:

    pairs = {}

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    rows = container.locator(
        "tr"
    )

    for i in range(
        rows.count()
    ):

        row = rows.nth(i)

        try:

            cells = unique_strings(
                row.locator(
                    "th, td"
                ).all_inner_texts()
            )

        except Exception:
            continue

        if len(cells) >= 2:

            pairs[
                cells[0]
            ] = " | ".join(
                cells[1:]
            )

    # --------------------------------------------------------
    # Definition lists
    # --------------------------------------------------------

    dts = container.locator(
        "dt"
    )

    for i in range(
        dts.count()
    ):

        dt = dts.nth(i)

        try:

            key = clean_text(
                dt.inner_text()
            )

            dd = dt.locator(
                "xpath=following-sibling::dd[1]"
            )

            if (
                key
                and dd.count()
            ):

                value = clean_text(
                    dd.first.inner_text()
                )

                if value:

                    pairs[key] = value

        except Exception:
            continue

    # --------------------------------------------------------
    # Generic 2-column rows
    # --------------------------------------------------------

    generic_rows = container.locator(
        "[role='row']"
    )

    for i in range(
        generic_rows.count()
    ):

        row = generic_rows.nth(i)

        try:

            cells = unique_strings(
                row.locator(
                    "[role='cell'], "
                    "[role='rowheader']"
                ).all_inner_texts()
            )

        except Exception:
            continue

        if len(cells) >= 2:

            pairs[
                cells[0]
            ] = " | ".join(
                cells[1:]
            )

    return pairs


# ============================================================
# GENERIC VISUAL ROW EXTRACTION
# ============================================================

def extract_visual_rows(
    container: Locator,
) -> list[dict]:
    """
    Fallback for Philips layouts that do not use semantic
    table markup.

    We preserve row text instead of guessing relationships.
    """

    rows = []

    selectors = [
        "li",
        "[class*='row']",
        "[class*='Row']",
        "[class*='item']",
        "[class*='Item']",
    ]

    seen = set()

    for selector in selectors:

        nodes = container.locator(
            selector
        )

        for i in range(
            nodes.count()
        ):

            node = nodes.nth(i)

            try:

                if not node.is_visible():
                    continue

                text = clean_text(
                    node.inner_text(
                        timeout=200
                    )
                )

            except Exception:
                continue

            if not text:
                continue

            if len(text) > 500:
                continue

            key = text.lower()

            if key in seen:
                continue

            seen.add(key)

            rows.append(
                {
                    "text": text
                }
            )

    return rows


# ============================================================
# TECHNICAL SPECIFICATION EXTRACTION
# ============================================================

def scrape_technical_specifications(
    page: Page,
) -> dict:

    result = {
        "found": False,
        "groups": {},
        "raw_text": None,
    }

    opened = (
        open_technical_specifications(
            page
        )
    )

    if not opened:

        print(
            "    Technical specifications "
            "not found."
        )

        return result

    dialog = find_technical_dialog(
        page
    )

    if dialog is None:

        print(
            "    Technical drawer not identified."
        )

        return result

    result["found"] = True

    # Force lazy-loaded content to appear.

    scroll_dialog_to_bottom(
        dialog
    )

    page.wait_for_timeout(
        300
    )

    scroll_dialog_to_top(
        dialog
    )

    # Expand every accessible group.

    expand_all_technical_groups(
        page,
        dialog,
    )

    # Scroll again after expansion.

    scroll_dialog_to_bottom(
        dialog
    )

    page.wait_for_timeout(
        300
    )

    # Preserve full raw evidence.

    try:

        result["raw_text"] = (
            clean_text(
                dialog.inner_text()
            )
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # ARIA accordion groups
    # --------------------------------------------------------

    controls = dialog.locator(
        "[aria-expanded]"
    )

    for i in range(
        controls.count()
    ):

        control = controls.nth(i)

        try:

            group_name = clean_text(
                control.inner_text(
                    timeout=200
                )
            )

        except Exception:
            continue

        if not group_name:
            continue

        if (
            group_name.lower()
            == "technical specifications"
        ):
            continue

        panel_id = (
            control.get_attribute(
                "aria-controls"
            )
        )

        if not panel_id:
            continue

        panel = dialog.locator(
            f"#{panel_id}"
        )

        if panel.count() == 0:
            continue

        try:

            panel_text = clean_text(
                panel.inner_text()
            )

        except Exception:
            continue

        if not panel_text:
            continue

        group_data = {
            "raw_text": panel_text
        }

        attributes = (
            extract_label_value_pairs(
                panel
            )
        )

        if attributes:

            group_data[
                "attributes"
            ] = attributes

        visual_rows = (
            extract_visual_rows(
                panel
            )
        )

        if visual_rows:

            group_data[
                "rows"
            ] = visual_rows

        result["groups"][
            group_name
        ] = group_data

    # --------------------------------------------------------
    # FALLBACK:
    #
    # If Philips doesn't expose aria-controls, preserve
    # visible accordion heading + surrounding content.
    # --------------------------------------------------------

    if not result["groups"]:

        headings = dialog.locator(
            "h2, h3, h4, h5"
        )

        for i in range(
            headings.count()
        ):

            heading = headings.nth(i)

            try:

                title = clean_text(
                    heading.inner_text()
                )

            except Exception:
                continue

            if not title:
                continue

            if (
                title.lower()
                == "technical specifications"
            ):
                continue

            try:

                sibling_text = (
                    heading.evaluate(
                        """
                        element => {

                            const values = [];

                            let node =
                                element.nextElementSibling;

                            while (node) {

                                if (
                                    /^H[2-5]$/.test(
                                        node.tagName
                                    )
                                ) {
                                    break;
                                }

                                const text =
                                    node.innerText?.trim();

                                if (text) {
                                    values.push(text);
                                }

                                node =
                                    node.nextElementSibling;
                            }

                            return values;
                        }
                        """
                    )
                )

            except Exception:

                sibling_text = []

            sibling_text = (
                unique_strings(
                    sibling_text
                )
            )

            if sibling_text:

                result["groups"][
                    title
                ] = {
                    "raw_text": " ".join(
                        sibling_text
                    ),
                    "items": sibling_text,
                }

    return result


# ============================================================
# DETAIL PAGE SCRAPER
# ============================================================

def scrape_product_detail(
    page: Page,
    product: PhilipsProduct,
) -> PhilipsProduct:

    if not product.product_url:

        product.scrape_errors.append(
            "No product URL found."
        )

        return product

    print(
        f"\n    Opening: "
        f"{product.product_url}"
    )

    try:

        page.goto(
            product.product_url,
            wait_until="domcontentloaded",
            timeout=PAGE_TIMEOUT_MS,
        )

    except PlaywrightTimeoutError:

        product.scrape_errors.append(
            "Detail page navigation timeout."
        )

        return product

    page.wait_for_timeout(
        WAIT_AFTER_NAVIGATION_MS
    )

    # --------------------------------------------------------
    # Lazy content
    # --------------------------------------------------------

    scroll_page(page)

    # --------------------------------------------------------
    # Show More sections
    # --------------------------------------------------------

    click_show_more_buttons(
        page
    )

    scroll_page(page)

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    h1 = page.locator("h1")

    if h1.count():

        try:

            product.detail_title = (
                clean_text(
                    h1.first.inner_text()
                )
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # JSON-LD
    # --------------------------------------------------------

    product.json_ld = (
        extract_json_ld(
            page
        )
    )

    # --------------------------------------------------------
    # Product metadata from page and JSON-LD
    # --------------------------------------------------------

    product.product_category = (
        extract_meta_value(
            page,
            [
                "meta[property='product:category']",
                "meta[name='category']",
            ],
        )
        or (
            next(
                (
                    value
                    for item in product.json_ld
                    if isinstance(item, dict)
                    for key, value in item.items()
                    if key.lower() in {"category", "productcategory"}
                    and value
                ),
                None,
            )
        )
    )

    product.breadcrumb_path = (
        extract_breadcrumbs(page)
    )

    product.product_features = (
        extract_product_features(page)
    )

    product.product_colors = (
        extract_product_colors(
            page,
            product.json_ld,
        )
    )

    product.rating_value, product.rating_count = (
        extract_rating_summary(page, product.json_ld)
    )

    product.availability_status = (
        extract_availability_status(page)
    )

    product.short_description = (
        extract_meta_value(
            page,
            [
                "meta[name='description']",
                "meta[property='og:description']",
                "meta[name='twitter:description']",
            ],
        )
    )

    product.detail_metadata = (
        extract_detail_metadata(
            page,
            product.json_ld,
        )
    )

    # --------------------------------------------------------
    # Page sections
    # --------------------------------------------------------

    product.detail_sections = (
        extract_page_sections(
            page
        )
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    try:

        product.detail_images = (
            extract_images(
                page.locator("body")
            )
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Links
    # --------------------------------------------------------

    product.detail_links = (
        extract_detail_links(
            page
        )
    )

    # --------------------------------------------------------
    # Technical specification drawer
    # --------------------------------------------------------

    try:

        product.technical_specifications = (
            scrape_technical_specifications(
                page
            )
        )

    except Exception as exc:

        product.scrape_errors.append(
            "Technical specification error: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # Raw page text
    #
    # Note: technical drawer evidence is stored separately.
    # --------------------------------------------------------

    try:

        body = page.locator(
            "body"
        )

        product.raw_detail_text = (
            clean_text(
                body.inner_text()
            )
        )

        product.long_description = (
            product.raw_detail_text
            or product.short_description
        )

    except Exception as exc:

        product.scrape_errors.append(
            f"Raw text error: {exc}"
        )

    product.scraped_at = now_utc()

    return product


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    products: list[PhilipsProduct],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / "checkpoint.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                asdict(product)
                for product in products
            ],
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# OUTPUT: FULL JSON
# ============================================================

def save_full_json(
    products: list[PhilipsProduct],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / "philips_products_full.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                asdict(product)
                for product in products
            ],
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nFull JSON: {path}"
    )


# ============================================================
# OUTPUT: JSONL
# ============================================================

def save_jsonl(
    products: list[PhilipsProduct],
) -> None:

    path = (
        OUTPUT_DIR
        / "philips_products.jsonl"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for product in products:

            file.write(
                json.dumps(
                    asdict(product),
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    print(
        f"JSONL: {path}"
    )


# ============================================================
# OUTPUT: FLATTENED CSV
# ============================================================

def save_flat_csv(
    products: list[PhilipsProduct],
) -> None:

    rows = []

    for product in products:

        if product.rating_value is None or product.rating_count is None:
            json_rating_value, json_rating_count = extract_aggregate_rating_from_json_ld(
                product.json_ld
            )
            if product.rating_value is None:
                product.rating_value = json_rating_value
            if product.rating_count is None:
                product.rating_count = json_rating_count

        row = {
            "brand": product.brand,
            "product_name": product.detail_title,
            "price_usd": product.price_usd,
            "product_category": product.product_category,
            "customer_rating":
                product.rating_value,

            "customer_rating_count":
                product.rating_count,

            "short_description":
                product.short_description,

            "long_description":
                product.long_description,

            "product_url":
                product.product_url,

            "listing_url":
                product.listing_url,

            "listing_features":
                json.dumps(
                    product.listing_features,
                    ensure_ascii=False,
                ),

            "image_urls":
                json.dumps(
                    product.image_urls,
                    ensure_ascii=False,
                ),

        }

        # Dynamically create columns from
        # listing attributes.

        for key, value in (
            product
            .listing_attributes
            .items()
        ):

            if isinstance(
                value,
                (dict, list),
            ):

                value = json.dumps(
                    value,
                    ensure_ascii=False,
                )

            row[
                f"listing_{key}"
            ] = value

        # Dynamically flatten technical
        # specification attributes.

        groups = (
            product
            .technical_specifications
            .get(
                "groups",
                {},
            )
        )

        for group_name, group in (
            groups.items()
        ):

            attributes = (
                group.get(
                    "attributes",
                    {}
                )
            )

            for key, value in (
                attributes.items()
            ):

                column = (
                    "spec__"
                    + group_name
                    + "__"
                    + key
                )

                column = re.sub(
                    r"\s+",
                    "_",
                    column.lower(),
                )

                row[column] = value

        rows.append(row)

    dataframe = pd.DataFrame(
        rows
    )

    dataframe = dataframe[['brand', 'product_name', 'price_usd', 'product_category', 'customer_rating', 'customer_rating_count', 'listing_app_connectivity',
                           'listing_plaque_claims', 'listing_gum_claims', 'listing_brush_modes_count', 'short_description', 'long_description', 'product_url']]

    # enforcing category to be Electric Toothbrush (as we are dealing with only electric toothbrushes in this scrapper)
    # design wise - It is already dynamic if we want to scrap other products as well
    dataframe["product_category"] = "Electric Toothbrush"

    dataframe.columns = dataframe.columns.str.replace(
        r"^listing_", "", regex=True
    )

    # Fill missing values with a placeholder to indicate that the data was not found in the sources.
    dataframe = dataframe.fillna("not found in sources")

    # normalziing the app_connectivity column to have "Yes" if the value is not "not found in sources", otherwise keep it as "not found in sources"
    dataframe["app_connectivity"] = dataframe["app_connectivity"].apply(
        lambda x: "Yes" if x != "not found in sources" else "not found in sources"
    )

    path = (
        OUTPUT_DIR
        / "philips_products_flat.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    print(
        f"Flat CSV: {path}"
    )


# ============================================================
# REPORT
# ============================================================

def print_report(
    products: list[PhilipsProduct],
) -> None:

    print(
        "\n"
        + "=" * 80
    )

    print("FINAL REPORT")

    print("=" * 80)

    print(
        f"Products discovered: "
        f"{len(products)}"
    )

    if (
        len(products)
        != EXPECTED_PRODUCTS
    ):

        print(
            "WARNING: "
            f"Expected approximately "
            f"{EXPECTED_PRODUCTS} "
            "in-stock products."
        )

    with_detail = sum(
        bool(
            product.raw_detail_text
        )
        for product in products
    )

    with_specs = sum(
        bool(
            product
            .technical_specifications
            .get(
                "found",
                False,
            )
        )
        for product in products
    )

    print(
        f"Detail pages scraped: "
        f"{with_detail}"
    )

    print(
        f"Technical specs found: "
        f"{with_specs}"
    )

    print()

    for index, product in enumerate(
        products,
        start=1,
    ):

        print(
            f"{index:02d}. "
            f"{product.product_name}"
        )

        print(
            "    Model:",
            ", ".join(
                product.model_codes
            ),
        )

        print(
            "    Price:",
            product.price_usd,
        )

        print(
            "    Listing features:",
            len(
                product.listing_features
            ),
        )

        print(
            "    Detail sections:",
            len(
                product.detail_sections
            ),
        )

        groups = (
            product
            .technical_specifications
            .get(
                "groups",
                {},
            )
        )

        print(
            "    Technical groups:",
            len(groups),
        )

        if product.scrape_errors:

            print(
                "    Errors:",
                product.scrape_errors,
            )


# ============================================================
# BROWSER SETUP
# ============================================================

def create_context(
    browser,
) -> BrowserContext:

    return browser.new_context(
        viewport={
            "width": 1600,
            "height": 1200,
        },

        locale="en-US",

        user_agent=(
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/128.0.0.0 "
            "Safari/537.36"
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    products = []

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=HEADLESS
        )

        context = create_context(
            browser
        )

        page = context.new_page()

        # ====================================================
        # STAGE 1:
        # Discover all products from listing pages.
        # ====================================================

        print(
            "\n"
            + "#" * 80
        )

        print(
            "STAGE 1 - PRODUCT DISCOVERY"
        )

        print(
            "#" * 80
        )

        max_pages = (
            TOTAL_LISTING_PAGES
            if TOTAL_LISTING_PAGES is not None
            else MAX_LISTING_PAGES
        )

        current_page = 1
        seen_pages = set()

        while current_page <= max_pages:

            if current_page in seen_pages:
                break

            seen_pages.add(current_page)

            try:

                page_products = (
                    scrape_listing_page(
                        page,
                        current_page,
                    )
                )

                products.extend(
                    page_products
                )

            except Exception as exc:

                print(
                    f"Listing page "
                    f"{current_page} failed: "
                    f"{exc}"
                )

            next_page = (
                find_next_listing_page(
                    page,
                    current_page,
                )
            )

            if next_page is None:
                break

            current_page = next_page

        products = (
            deduplicate_products(
                products
            )
        )

        print(
            f"\nTotal unique products "
            f"discovered: "
            f"{len(products)}"
        )

        save_checkpoint(
            products
        )

        # ====================================================
        # STAGE 2:
        # Visit every product detail page.
        # ====================================================

        print(
            "\n"
            + "#" * 80
        )

        print(
            "STAGE 2 - DETAIL ENRICHMENT"
        )

        print(
            "#" * 80
        )

        total = len(products)

        for index, product in enumerate(
            products,
            start=1,
        ):

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"[{index}/{total}] "
                f"{product.product_name}"
            )

            try:

                products[
                    index - 1
                ] = scrape_product_detail(
                    page,
                    product,
                )

            except Exception as exc:

                product.scrape_errors.append(
                    f"Unhandled detail error: "
                    f"{exc}"
                )

            # Save after every product.
            #
            # If product 25 crashes, you still retain
            # everything scraped for products 1-24.

            save_checkpoint(
                products
            )

            # Conservative request pacing.
            time.sleep(0.5)

        context.close()

        browser.close()

    # ========================================================
    # OUTPUT
    # ========================================================

    save_full_json(
        products
    )

    save_jsonl(
        products
    )

    save_flat_csv(
        products
    )

    print_report(
        products
    )


if __name__ == "__main__":
    main()