# ---------------------------------------------
# src/scraper.py — Bank of Maharashtra Loan Scraper
# ---------------------------------------------
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
import time

# -----------------------
# Project-relative paths
# -----------------------
BASE_DIR = Path(__file__).parent.parent  # src/ -> project root
RAW_DIR = BASE_DIR / "data/raw"
PROC_DIR = BASE_DIR / "data/processed"
META_FILE = BASE_DIR / "data/scrape_metadata.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# URLs & headers
# -----------------------
BASE_URLS = [
    "https://bankofmaharashtra.bank.in/personal-banking/loans/car-loan",
    "https://bankofmaharashtra.bank.in/personal-banking/loans/home-loan",
    "https://bankofmaharashtra.bank.in/gold-loan",
    "https://bankofmaharashtra.bank.in/personal-banking/loans/personal-loan",
    "https://bankofmaharashtra.bank.in/retail-interest-rates",
    "https://bankofmaharashtra.bank.in/educational-loans",
    "https://bankofmaharashtra.bank.in/topup-home-loan",
    "https://bankofmaharashtra.bank.in/advances"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (LoanAssistantBot)"}

# -----------------------
# Cleaner function
# -----------------------
def extract_main_content(soup):
    """Extract loan content reliably from BoM website."""
    candidates = []

    POSSIBLE_SELECTORS = [
        {"tag": "div", "attrs": {"class": "col-md-8"}},
        {"tag": "div", "attrs": {"class": "tabcontent"}},
        {"tag": "div", "attrs": {"class": "service-content"}},
        {"tag": "div", "attrs": {"class": "container"}},
        {"tag": "section"},
        {"tag": "article"},
    ]

    for sel in POSSIBLE_SELECTORS:
        candidates.extend(soup.find_all(sel.get("tag"), sel.get("attrs")))

    if not candidates:
        candidates = [soup.body]

    text_blocks = []
    for block in candidates:
        for tag in block.find_all(["p", "li", "h1", "h2", "h3", "table"]):
            cleaned = tag.get_text(" ", strip=True)
            if len(cleaned) > 5:
                text_blocks.append(cleaned)

    # Remove duplicates and join
    return "\n".join(list(dict.fromkeys(text_blocks)))

# -----------------------
# Scraper function
# -----------------------
def scrape_page(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    cleaned_text = extract_main_content(soup)
    return r.text, cleaned_text

# -----------------------
# Main runner
# -----------------------
def run_scraper():
    metadata = []

    for url in BASE_URLS:
        try:
            html, clean = scrape_page(url)
            fname = url.split("/")[-1].replace("/", "") or "index"
            raw_file = RAW_DIR / f"{fname}.html"
            txt_file = PROC_DIR / f"{fname}.txt"

            raw_file.write_text(html, encoding="utf-8")
            txt_file.write_text(clean, encoding="utf-8")

            metadata.append({
                "url": url,
                "raw": str(raw_file),
                "clean": str(txt_file),
                "chars": len(clean),
                "timestamp": time.time(),
            })

            time.sleep(1)  # polite scraping

        except Exception as e:
            print(f"[ERROR] Failed scraping {url}: {e}")

    META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("✅ Scraping completed successfully.")

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    run_scraper()
