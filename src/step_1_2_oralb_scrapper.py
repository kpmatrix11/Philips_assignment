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


ORALB_DOMAIN = "https://oralb.com"
LISTING_URL = "https://oralb.com/en-us/products/electric-toothbrushes/electric-toothbrushes/"
OUTPUT_DIR = Path("data/oralb")
PAGE_TIMEOUT_MS = 60_000
WAIT_AFTER_NAVIGATION_MS = 2_000
WAIT_AFTER_CLICK_MS = 500
HEADLESS = True


@dataclass
class OralBProduct:
    brand: str = "Oral-B"
    product_name: Optional[str] = None
    product_type: Optional[str] = None
    model_codes: list[str] = field(default_factory=list)
    price_usd: Optional[float] = None
    listing_features: list[str] = field(default_factory=list)
    listing_attributes: dict = field(default_factory=dict)
    image_urls: list[str] = field(default_factory=list)
    listing_url: Optional[str] = None
    product_url: Optional[str] = None
    raw_listing_text: Optional[str] = None
    detail_title: Optional[str] = None
    product_category: Optional[str] = None
    breadcrumb_path: list[str] = field(default_factory=list)
    availability_status: Optional[str] = None
    short_description: Optional[str] = None
    long_description: Optional[str] = None
    app_connectivity: Optional[bool] = None
    product_features: list[str] = field(default_factory=list)
    product_colors: list[str] = field(default_factory=list)
    rating_value: Optional[float] = None
    rating_count: Optional[int] = None
    product_variants: list[dict] = field(default_factory=list)
    json_ld: list[dict] = field(default_factory=list)
    detail_sections: dict = field(default_factory=dict)
    technical_specifications: dict = field(default_factory=dict)
    detail_links: list[dict] = field(default_factory=list)
    detail_images: list[str] = field(default_factory=list)
    raw_detail_text: Optional[str] = None
    detail_metadata: dict = field(default_factory=dict)
    scraped_at: Optional[str] = None
    scrape_errors: list[str] = field(default_factory=list)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
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


def parse_price(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"\$\s*([\d,]+(?:\.\d{1,2})?)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def extract_listing_price(page: Page, product_name: Optional[str], product_url: Optional[str] = None) -> Optional[float]:
    direct_price = parse_price(product_name)
    if direct_price is not None:
        return direct_price

    body_text = clean_text(page.locator("body").inner_text(timeout=3000)) or ""
    if not body_text:
        return None

    candidates: list[str] = []
    if product_name:
        normalized = re.sub(r"^oral-b\s+", "", product_name, flags=re.I)
        candidates.extend([
            product_name,
            normalized,
            re.sub(r"\s+", " ", normalized).strip(),
        ])

    if product_url:
        slug = product_url.split("/products/")[-1].split("/")[-1]
        if slug:
            candidates.append(slug.replace("-", " "))

    seen = set()
    for candidate in candidates:
        cleaned = clean_text(candidate)
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())

        # First try the exact product title/slug in the page body.
        exact_pattern = rf"{re.escape(cleaned)}.*?\$\s*(\d[\d,]*(?:\.\d{{1,2}})?)"
        match = re.search(exact_pattern, body_text, flags=re.I | re.S)
        if match:
            return float(match.group(1).replace(",", ""))

        # Fallback for product variants whose title differs slightly from the rendered listing title.
        series_match = re.search(r"(?:iO|oral-b).*?series\s*\d+", cleaned, flags=re.I)
        if series_match:
            prefix = series_match.group(0)
            prefix_pattern = rf"{re.escape(prefix)}.*?\$\s*(\d[\d,]*(?:\.\d{{1,2}})?)"
            match = re.search(prefix_pattern, body_text, flags=re.I | re.S)
            if match:
                return float(match.group(1).replace(",", ""))

        # Last fallback: search the page body for the first nearby money value after the product's significant keywords.
        keywords = re.findall(r"(?:series\s*\d+|twin\s*pack|starter\s*kit|genius|rechargeable\s+electric\s+toothbrush|electric\s+toothbrush|customizable\s+clean)", cleaned, flags=re.I)
        for keyword in keywords:
            keyword_pattern = rf"{re.escape(keyword)}.*?\$\s*(\d[\d,]*(?:\.\d{{1,2}})?)"
            match = re.search(keyword_pattern, body_text, flags=re.I | re.S)
            if match:
                return float(match.group(1).replace(",", ""))

    # Final fallback: if the page still shows a price block near the product area, grab the first one.
    generic_match = re.search(r"(?:iO|oral-b).*?(?:series\s*\d+|twin\s*pack|starter\s*kit|genius).*?\$\s*(\d[\d,]*(?:\.\d{{1,2}})?)", body_text, flags=re.I | re.S)
    if generic_match:
        return float(generic_match.group(1).replace(",", ""))

    return None


def absolute_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    return urljoin(ORALB_DOMAIN, url)


def normalize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    normalized = url.strip()
    if not normalized:
        return None
    normalized = normalized.split("#", 1)[0].split("?", 1)[0]
    normalized = normalized.rstrip("/")
    return normalized.lower() or None


def extract_model_codes(text: Optional[str]) -> list[str]:
    if not text:
        return []
    patterns = [
        r"\b(?:IO|iO)[A-Z0-9-]+\b",
        r"\bSeries\s*[0-9]+\b",
        r"\b(?:GENIUS|PRO|SMART|KIDS|MAX|PLUS|LITE)[A-Z0-9-]*\b",
        r"\b[A-Z]{2,6}[0-9]{2,6}(?:/[A-Z0-9-]+)?\b",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.I))
    cleaned = [item.strip() for item in found if item.strip()]
    normalized = []
    for item in cleaned:
        value = item.strip()
        if value.lower().startswith("series"):
            normalized.append(value.upper())
        else:
            normalized.append(value.upper())
    return unique_strings(normalized)


def extract_json_ld(page: Page) -> list[dict]:
    output = []
    scripts = page.locator("script[type='application/ld+json']")
    for i in range(scripts.count()):
        try:
            raw = scripts.nth(i).text_content()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, dict):
                output.append(data)
            elif isinstance(data, list):
                output.extend(item for item in data if isinstance(item, dict))
        except Exception:
            continue
    return output


def extract_aggregate_rating_from_json_ld(json_ld: list[dict]) -> tuple[Optional[float], Optional[int]]:
    def walk(node):
        if isinstance(node, dict):
            if "aggregateRating" in node:
                return node["aggregateRating"]
            for value in node.values():
                result = walk(value)
                if result is not None:
                    return result
        elif isinstance(node, list):
            for value in node:
                result = walk(value)
                if result is not None:
                    return result
        return None

    aggregate = walk(json_ld)
    if not isinstance(aggregate, dict):
        return None, None
    rating_value = aggregate.get("ratingValue")
    review_count = aggregate.get("reviewCount", aggregate.get("ratingCount"))
    try:
        rating = float(rating_value) if rating_value is not None else None
    except (TypeError, ValueError):
        rating = None
    try:
        count = int(float(review_count)) if review_count is not None else None
    except (TypeError, ValueError):
        count = None
    return rating, count


def extract_rating_summary(page: Page, json_ld: Optional[list[dict]] = None) -> tuple[Optional[float], Optional[int]]:
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
            if not value:
                continue
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
            if not value:
                continue
            try:
                count = int(float(value))
                break
            except Exception:
                pass
        if count is not None:
            break

    return rating, count


def extract_meta_value(page: Page, selectors: list[str]) -> Optional[str]:
    for selector in selectors:
        loc = page.locator(selector)
        for i in range(loc.count()):
            value = loc.nth(i).get_attribute("content")
            if value:
                return clean_text(value)
    return None


def extract_breadcrumbs(page: Page) -> list[str]:
    crumbs: list[str] = []
    selectors = [
        "nav a",
        "[class*='breadcrumb'] a",
        "[aria-label*='breadcrumb' i] a",
        "[itemprop='itemListElement'] a",
    ]
    for selector in selectors:
        nodes = page.locator(selector)
        for i in range(nodes.count()):
            try:
                text = clean_text(nodes.nth(i).inner_text(timeout=200))
            except Exception:
                continue
            if text and text.lower() not in {"home", "products"}:
                crumbs.append(text)
    return unique_strings(crumbs)


def extract_product_colors(page: Page, json_ld: Optional[list[dict]] = None) -> list[str]:
    colors: list[str] = []
    if json_ld:
        def walk(node):
            if isinstance(node, dict):
                for key in ["color", "colors", "colorOption", "colorOptions", "colorName"]:
                    value = node.get(key)
                    if isinstance(value, str):
                        colors.extend([part.strip() for part in re.split(r"[|,/]+", value) if part.strip()])
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                colors.extend([part.strip() for part in re.split(r"[|,/]+", item) if part.strip()])
                            elif isinstance(item, dict):
                                walk(item)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
        walk(json_ld)

    locators = [
        page.locator("[aria-label*='color' i]"),
        page.locator("[data-testid*='color' i]"),
        page.locator("[data-color]"),
        page.locator("[class*='color' i]"),
        page.locator("button[title]"),
        page.locator("button[aria-label]"),
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
            bits = [part.strip() for part in re.split(r"[|,/]+", text) if part.strip()]
            for part in bits:
                if any(word in part.lower() for word in [
                    "black","white","blue","silver","gold","pink","purple","gray","rose","red","green","brown","beige","ivory","navy","graphite","champagne","charcoal","starlight","midnight","glacier","violet","white" ]):
                    colors.append(part)
    return unique_strings(colors)


def extract_page_sections(page: Page) -> dict:
    result: dict[str, list[str]] = {}
    headings = page.locator("h2, h3, h4")
    for i in range(headings.count()):
        heading = headings.nth(i)
        try:
            if not heading.is_visible():
                continue
            title = clean_text(heading.inner_text(timeout=300))
        except Exception:
            continue
        if not title:
            continue
        try:
            content = heading.evaluate("""
                element => {
                  const output = [];
                  let node = element.nextElementSibling;
                  while (node) {
                    if (['H2','H3','H4'].includes(node.tagName)) break;
                    const text = node.innerText?.trim();
                    if (text) output.push(text);
                    node = node.nextElementSibling;
                  }
                  return output;
                }
            """)
        except Exception:
            content = []
        cleaned = unique_strings(content)
        if not cleaned:
            continue
        key = title
        counter = 2
        while key in result:
            key = f"{title} ({counter})"
            counter += 1
        result[key] = cleaned
    return result


def extract_more_information_text(page: Page) -> Optional[str]:
    selectors = [
        'a:has-text("More Information")',
        'button:has-text("More Information")',
        'text=More Information',
        'text=More information',
    ]
    for selector in selectors:
        try:
            links = page.locator(selector)
        except Exception:
            continue
        for i in range(links.count()):
            link = links.nth(i)
            try:
                label = clean_text(link.inner_text(timeout=300))
            except Exception:
                continue
            if not label or "more information" not in label.lower():
                continue
            try:
                section_text = link.evaluate("""
                    element => {
                        const roots = [
                            element.closest('li'),
                            element.closest('section'),
                            element.closest('article'),
                            element.closest('div'),
                            element.parentElement,
                        ].filter(Boolean);

                        for (const root of roots) {
                            const text = (root.innerText || '').trim();
                            if (!text) continue;
                            if (/what's in the box|make the most of every session|warranty information/i.test(text)) {
                                return text;
                            }
                        }

                        let node = element.parentElement;
                        while (node) {
                            const text = (node.innerText || '').trim();
                            if (text && /what's in the box|make the most of every session|warranty information/i.test(text)) {
                                return text;
                            }
                            node = node.parentElement;
                        }
                        return element.innerText || '';
                    }
                """)
            except Exception:
                section_text = ""

            cleaned = clean_text(section_text)
            if cleaned:
                cleaned = re.sub(r"\s+", " ", cleaned)
                return cleaned
    return None


def extract_product_features(page: Page) -> list[str]:
    features: list[str] = []
    locators = [page.locator("li"), page.locator("[class*='feature']"), page.locator("[class*='benefit']")]
    for loc in locators:
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


def extract_more_information_from_features(product_features: list[str]) -> Optional[str]:
    if not product_features:
        return None
    start_index = None
    end_index = None
    for index, feature in enumerate(product_features):
        if "more information" in feature.lower():
            start_index = index + 1
            break
    if start_index is None:
        return None
    for index in range(start_index, len(product_features)):
        feature = product_features[index]
        if "what's in the box" in feature.lower() or "what is in the box" in feature.lower():
            end_index = index
            break
    if end_index is None:
        end_index = len(product_features)
    slice_items = product_features[start_index:end_index]
    if not slice_items:
        return None
    text = " ".join(slice_items)
    return clean_text(text) or None


def extract_app_connectivity_flag(product_features: list[str], long_description: Optional[str] = None) -> bool:
    text = " ".join(product_features or [])
    if long_description:
        text = f"{text} {long_description}"
    lower = text.lower()
    return any(keyword in lower for keyword in [
        "connect to the oral-b app",
        "oral-b app",
        "bluetooth",
        "app using bluetooth",
        "mobile app",
        "smart charger",
    ])


def extract_claims_and_brush_mode_count(description: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[int]]:
    if not description:
        return None, None, None

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\s*;\s*", description)
        if sentence.strip()
    ]
    plaque_claims = unique_strings([
        sentence for sentence in sentences
        if re.search(r"\bplaque\b", sentence, flags=re.I)
    ])
    gum_claims = unique_strings([
        sentence for sentence in sentences
        if re.search(r"\b(?:gum|gums|gumline|gingiva|gingival)\b", sentence, flags=re.I)
    ])

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    count_pattern = re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:(?:different|smart|unique|available|total|cleaning|brushing|customi[sz]able|personalized|personalised)\s+)*"
        r"(?:brushing\s+)?modes?\b",
        flags=re.I,
    )
    mode_match = count_pattern.search(description)
    mode_count = None
    if mode_match:
        count_text = mode_match.group(1).lower()
        mode_count = int(count_text) if count_text.isdigit() else number_words[count_text]

    return (
        " | ".join(plaque_claims) or None,
        " | ".join(gum_claims) or None,
        mode_count,
    )


def extract_availability_status(page: Page) -> Optional[str]:
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


def extract_detail_links(page: Page) -> list[dict]:
    output = []
    seen = set()
    links = page.locator("a[href]")
    for i in range(links.count()):
        link = links.nth(i)
        try:
            href = link.get_attribute("href")
            if not href:
                continue
            url = absolute_url(href)
            text = clean_text(link.inner_text(timeout=200))
            key = (text, url)
            if key in seen:
                continue
            seen.add(key)
            output.append({"text": text, "url": url})
        except Exception:
            continue
    return output


def extract_images(container: Locator) -> list[str]:
    images = []
    nodes = container.locator("img")
    for i in range(nodes.count()):
        image = nodes.nth(i)
        for attr in ["src", "data-src", "data-lazy-src"]:
            value = image.get_attribute(attr)
            if value:
                images.append(absolute_url(value))
        srcset = image.get_attribute("srcset")
        if srcset:
            for item in srcset.split(","):
                candidate = item.strip().split(" ")[0]
                if candidate:
                    images.append(absolute_url(candidate))
    return unique_strings(images)


def find_product_links(page: Page) -> list[dict]:
    anchor_map: dict[str, dict[str, str]] = {}
    anchors = page.locator("a[href]")

    for i in range(anchors.count()):
        link = anchors.nth(i)
        try:
            href = link.get_attribute("href")
            if not href:
                continue

            text = clean_text(link.inner_text(timeout=200))
            full_url = absolute_url(href)
            if not full_url:
                continue

            normalized_url = normalize_url(full_url)
            if not normalized_url:
                continue

            if "/products/electric-toothbrushes/" not in href and "/products/electric-toothbrushes" not in href:
                continue

            if normalized_url in {
                "https://oralb.com/en-us/products/electric-toothbrushes",
                "https://oralb.com/en-us/products/electric-toothbrushes/",
                "https://oralb.com/en-us/products/electric-toothbrushes/electric-toothbrushes",
                "https://oralb.com/en-us/products/electric-toothbrushes/electric-toothbrushes/",
            }:
                continue

            card_text = clean_text(link.evaluate("""
                (element) => {
                    const root = element.closest('article, li, div, section');
                    return root ? root.innerText : element.innerText;
                }
            """))
            card_source = card_text or text or ""
            if not card_source:
                continue

            title_lower = card_source.lower()
            low = (title_lower + " " + href.lower())

            # Avoid promo, category, and landing links masquerading as product cards.
            if any(keyword in title_lower for keyword in [
                "top deals",
                "grab the",
                "before they’re gone",
                "before they are gone",
                "save up",
                "learn more",
                "explore more",
                "shop all",
                "gift guide",
                "wedding gift guide",
                "replacement brush heads",
                "manual toothbrushes",
                "battery toothbrushes",
                "water flosser",
                "electric toothbrushes",
                "kids toothbrushes",
                "smart series",
                "genius series",
                "bundles and electric toothbrushes",
                "twin packs and bundles",
                "home",
                "shop series",
            ]):
                continue

            product_pattern = re.compile(
                r"(?:io|oral-b io|oralb io).*?(?:series\s*[0-9]|genius|customizable clean|starter kit)|"
                r"(?:series\s*[0-9].*(?:electric toothbrush|rechargeable electric toothbrush|customizable clean|genius))|"
                r"io[0-9]|i\s*o\s*series\s*[0-9]",
                re.I,
            )
            if not product_pattern.search(low):
                continue

            # Ignore collection root and category navigation links.
            if normalized_url.endswith("/products/electric-toothbrushes") or normalized_url.endswith("/products/electric-toothbrushes/"):
                continue

            if any(token in normalized_url for token in [
                "/products/collections/",
                "/products/category/",
                "/products/twin-packs-and-bundles/",
                "/products/genius-series/",
                "/products/smart-series/",
                "/products/pro-series/",
            ]):
                continue

            candidate_text = (text or card_source).strip()
            if len(candidate_text.split()) < 3:
                continue

            anchor_map.setdefault(normalized_url, {"text": candidate_text, "card_text": card_source or candidate_text})
        except Exception:
            continue

    return [{"url": url, "text": data["text"], "card_text": data["card_text"]} for url, data in anchor_map.items()]


def derive_listing_attributes(features: list[str]) -> dict:
    attrs: dict[str, Any] = {}
    for feature in features:
        lower = feature.lower()
        match = re.search(r"(\d+)\s+brushing\s+modes?", lower)
        if match:
            attrs["brushing_modes_count"] = int(match.group(1))
        match = re.search(r"(\d+)\s+cleaning\s+modes?", lower)
        if match:
            attrs["cleaning_modes_count"] = int(match.group(1))
        match = re.search(r"(\d+)\s+minute(?:s)?", lower)
        if match:
            attrs["timed_cleaning_minutes"] = int(match.group(1))
        if "pressure sensor" in lower:
            attrs["pressure_sensor"] = feature
        if "app" in lower or "bluetooth" in lower:
            attrs["app_connectivity"] = feature
        if "battery" in lower:
            attrs["battery"] = feature
        if "smart" in lower:
            attrs["smart_features"] = feature
    return attrs


def extract_listing_features(card: Locator) -> list[str]:
    features: list[str] = []
    list_items = card.locator("li")
    for i in range(list_items.count()):
        try:
            value = clean_text(list_items.nth(i).inner_text(timeout=300))
        except Exception:
            continue
        if value and len(value) <= 350:
            features.append(value)
    return unique_strings(features)


def extract_listing_card(card: Locator, listing_url: str) -> Optional[OralBProduct]:
    try:
        raw_text = clean_text(card.inner_text(timeout=1000))
    except Exception:
        return None
    if not raw_text:
        return None
    model_codes = extract_model_codes(raw_text)
    if not model_codes:
        model_codes = extract_model_codes(listing_url)
    if not model_codes:
        return None

    product_name = None
    for candidate in [card.locator("h1,h2,h3,h4").all_inner_texts(), raw_text.splitlines()]:
        for item in candidate:
            text = clean_text(item)
            if text and any(word in text.lower() for word in ["oral-b", "oralb", "io", "series", "electric toothbrush", "toothbrush"]):
                product_name = text
                break
        if product_name:
            break

    product_url = None
    links = card.locator("a[href]")
    for i in range(links.count()):
        href = links.nth(i).get_attribute("href")
        if href and "/products/" in href:
            product_url = absolute_url(href)
            break

    features = extract_listing_features(card)
    return OralBProduct(
        product_name=product_name or raw_text[:200],
        product_type=("Electric toothbrush" if "toothbrush" in raw_text.lower() else None),
        model_codes=model_codes,
        price_usd=parse_price(raw_text),
        listing_features=features,
        listing_attributes=derive_listing_attributes(features),
        image_urls=extract_images(card),
        listing_url=listing_url,
        product_url=product_url,
        raw_listing_text=raw_text,
        scraped_at=now_utc(),
    )


def product_key(product: OralBProduct) -> Optional[str]:
    if product.product_url:
        return product.product_url
    if product.model_codes:
        return product.model_codes[0]
    if product.product_name:
        return product.product_name.lower()
    return None


def product_richness_score(product: OralBProduct) -> int:
    return len(product.listing_features) + len(product.listing_attributes) + len(product.image_urls) + (5 if product.product_url else 0) + (3 if product.product_name else 0)


def deduplicate_products(products: list[OralBProduct]) -> list[OralBProduct]:
    output: dict[str, OralBProduct] = {}
    for product in products:
        key = product_key(product)
        if not key:
            continue
        existing = output.get(key)
        if existing is None:
            output[key] = product
            continue
        if product_richness_score(product) > product_richness_score(existing):
            output[key] = product
    return list(output.values())


def scrape_listing_page(page: Page) -> list[OralBProduct]:
    print("\n" + "=" * 80)
    print("LISTING PAGE")
    print(LISTING_URL)
    page.goto(LISTING_URL, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    page.wait_for_timeout(WAIT_AFTER_NAVIGATION_MS)

    links = find_product_links(page)
    print(f"Candidate product links: {len(links)}")
    products: list[OralBProduct] = []
    for index, item in enumerate(links, start=1):
        url = item["url"]
        text = item["text"]
        card_text = item.get("card_text") or text
        price = parse_price(text) or parse_price(card_text) or extract_listing_price(page, text, url)
        candidate = OralBProduct(
            product_name=text or "Oral-B product",
            product_type="Electric toothbrush",
            model_codes=extract_model_codes(f"{text} {url}"),
            price_usd=price,
            listing_url=LISTING_URL,
            product_url=url,
            raw_listing_text=text,
            scraped_at=now_utc(),
        )
        if not candidate.model_codes:
            candidate.model_codes = ["ORALB"]
        products.append(candidate)
        print(f"  {index:02d}. {candidate.product_name} {candidate.model_codes} ${candidate.price_usd}")
    return deduplicate_products(products)


def extract_detail_metadata(page: Page, json_ld: list[dict]) -> dict:
    metadata: dict[str, Any] = {}
    description = extract_meta_value(page, ["meta[name='description']", "meta[property='og:description']", "meta[name='twitter:description']"])
    rating_value, rating_count = extract_rating_summary(page, json_ld)
    if description:
        metadata["description"] = description
    metadata["breadcrumbs"] = extract_breadcrumbs(page)
    metadata["availability"] = extract_availability_status(page)
    metadata["rating_summary"] = {"rating_value": rating_value, "rating_count": rating_count}
    for item in json_ld:
        if not isinstance(item, dict):
            continue
        for key in ["brand", "category", "sku", "gtin", "mpn", "productID", "availability", "price", "priceCurrency", "url"]:
            value = item.get(key)
            if value not in (None, ""):
                metadata.setdefault("json_ld", {})[key] = value
    return metadata


def scrape_product_detail(page: Page, product: OralBProduct) -> OralBProduct:
    if not product.product_url:
        product.scrape_errors.append("No product URL found.")
        return product
    print(f"\n    Opening: {product.product_url}")
    try:
        page.goto(product.product_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        product.scrape_errors.append("Detail page navigation timeout.")
        return product
    page.wait_for_timeout(WAIT_AFTER_NAVIGATION_MS)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(1000)

    h1 = page.locator("h1")
    if h1.count():
        try:
            product.detail_title = clean_text(h1.first.inner_text())
        except Exception:
            pass

    product.json_ld = extract_json_ld(page)
    product.product_category = extract_meta_value(page, ["meta[property='product:category']", "meta[name='category']"]) or next((value for item in product.json_ld if isinstance(item, dict) for key, value in item.items() if isinstance(value, str) and key.lower() in {"category", "productcategory"}), None)
    product.breadcrumb_path = extract_breadcrumbs(page)
    product.product_features = extract_product_features(page)
    product.product_colors = extract_product_colors(page, product.json_ld)
    product.rating_value, product.rating_count = extract_rating_summary(page, product.json_ld)
    product.availability_status = extract_availability_status(page)
    product.short_description = extract_meta_value(page, ["meta[name='description']", "meta[property='og:description']", "meta[name='twitter:description']"])
    product.detail_metadata = extract_detail_metadata(page, product.json_ld)
    product.detail_sections = extract_page_sections(page)
    try:
        product.detail_images = extract_images(page.locator("body"))
    except Exception:
        pass
    product.detail_links = extract_detail_links(page)

    try:
        body = page.locator("body")
        product.raw_detail_text = clean_text(body.inner_text())
        product.long_description = (
            extract_more_information_from_features(product.product_features)
            or extract_more_information_text(page)
            or product.raw_detail_text
            or product.short_description
        )
        product.app_connectivity = extract_app_connectivity_flag(product.product_features, product.long_description)
    except Exception as exc:
        product.scrape_errors.append(f"Raw text error: {exc}")

    product.scraped_at = now_utc()
    return product


def save_checkpoint(products: list[OralBProduct]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "checkpoint.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump([asdict(product) for product in products], file, ensure_ascii=False, indent=2)


def save_full_json(products: list[OralBProduct]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "oralb_products_full.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump([asdict(product) for product in products], file, ensure_ascii=False, indent=2)
    print(f"\nFull JSON: {path}")


def save_jsonl(products: list[OralBProduct]) -> None:
    path = OUTPUT_DIR / "oralb_products.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for product in products:
            file.write(json.dumps(asdict(product), ensure_ascii=False))
            file.write("\n")
    print(f"JSONL: {path}")


def save_flat_csv(products: list[OralBProduct]) -> None:
    rows = []
    for product in products:
        plaque_claim, gum_claims, brush_mode_count = extract_claims_and_brush_mode_count(product.long_description)
        row = {
            "brand": product.brand,
            "product_name": product.detail_title,
            # "model_codes": " | ".join(product.model_codes),
            "price_usd": product.price_usd,
            "product_category": product.product_category,
            "customer_rating": product.rating_value,
            "customer_rating_count": product.rating_count,
            "app_connectivity": product.app_connectivity,
            "short_description": product.short_description,
            "long_description": product.long_description,
            "plaque_claims": plaque_claim,
            "gum_claims": gum_claims,
            "brush_modes_count": brush_mode_count,
            "product_url": product.product_url,
            # "listing_url": product.listing_url,
            # "listing_features": json.dumps(product.listing_features, ensure_ascii=False),
            # "image_urls": json.dumps(product.image_urls, ensure_ascii=False),
            
            # "scrape_errors": json.dumps(product.scrape_errors, ensure_ascii=False),
        }
        for key, value in product.listing_attributes.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            row[f"listing_{key}"] = value
        rows.append(row)
    dataframe = pd.DataFrame(rows)

    # enforcing category to be Electric Toothbrush (as we are dealing with only electric toothbrushes in this scrapper)
    # design wise - It is already dynamic if we want to scrap other products as well
    dataframe["product_category"] = "Electric Toothbrush"

    # Fill missing values with a placeholder to indicate that the data was not found in the sources.
    dataframe = dataframe.fillna("not found in sources")

    # normalziing the app_connectivity column to have "Yes" if the value is not "not found in sources", otherwise keep it as "not found in sources"
    dataframe["app_connectivity"] = dataframe["app_connectivity"].apply(
        lambda x: "Yes" if str(x).upper() == "TRUE"
        else "not found in sources"
    )

    path = OUTPUT_DIR / "oralb_products_flat.csv"
    dataframe.to_csv(path, index=False)
    print(f"Flat CSV: {path}")


def print_report(products: list[OralBProduct]) -> None:
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(f"Products discovered: {len(products)}")
    for index, product in enumerate(products, start=1):
        print(f"{index:02d}. {product.product_name}")
        print("    Model:", ", ".join(product.model_codes))
        print("    Price:", product.price_usd)
        print("    Listing features:", len(product.listing_features))
        print("    Detail sections:", len(product.detail_sections))
        if product.scrape_errors:
            print("    Errors:", product.scrape_errors)


def create_context(browser) -> BrowserContext:
    return browser.new_context(
        viewport={"width": 1600, "height": 1200},
        locale="en-US",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    products: list[OralBProduct] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = create_context(browser)
        page = context.new_page()
        try:
            listing_products = scrape_listing_page(page)
            products.extend(listing_products)
            products = deduplicate_products(products)
            save_checkpoint(products)
            total = len(products)
            for index, product in enumerate(products, start=1):
                print(f"\n" + "-" * 80)
                print(f"[{index}/{total}] {product.product_name}")
                try:
                    products[index - 1] = scrape_product_detail(page, product)
                except Exception as exc:
                    product.scrape_errors.append(f"Unhandled detail error: {exc}")
                save_checkpoint(products)
                time.sleep(0.5)
        finally:
            context.close()
            browser.close()

    save_full_json(products)
    save_jsonl(products)
    save_flat_csv(products)
    print_report(products)


if __name__ == "__main__":
    main()
