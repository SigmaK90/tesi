import requests
import time
import pandas as pd
from keybert import KeyBERT
from collections import Counter

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

def fetch_open_markets(limit=500):
    response = requests.get(
        url=GAMMA_MARKETS_URL,
        params={
            "active":"true",
            "closed":"false",
            "limit": limit
        },
        timeout= 20
    )
    response.raise_for_status()
    return response.json()

def fetch_open_markets_sample(max_pages=4):
    markets = []
    offset = 0
    page_limit = 500

    for page in range(max_pages):
        print(f"Fetching page {page+1}/{max_pages}")

        response = requests.get(
            GAMMA_MARKETS_URL,
            params={
                "active": "true",
                "closed": "false",
                "limit": page_limit,
                "offset": offset
            },
            timeout=30
        )

        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        markets.extend(batch)

        offset += page_limit

    return markets

def normalize_markets(raw_market):
    return {
        "id": str(raw_market.get("id", "")),
        "question": str(raw_market.get("question", "")).strip(),
        "description": str(raw_market.get("description", "")).strip(),
        "conditionId": str(raw_market.get("conditionId", "")).strip(),
        "liquidity": float(raw_market.get("liquidity", 0) or 0),
        "volume": float(raw_market.get("volume", 0) or 0),
        "end_date": str(raw_market.get("endDate", "")).strip(),
    }

raw_markets = fetch_open_markets_sample()
print(len(raw_markets))

markets = [normalize_markets(m) for m in raw_markets]

df = pd.DataFrame(markets)

df["full_text"] = (
    df["question"].fillna("") + " " + df["description"].fillna("")
).str.lower().str.strip()

kw_model = KeyBERT()

sample_text = df.loc[0, "question"]

def extract_market_keywords(text, top_n=5):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )
    return [kw[0] for kw in keywords]

sample_df = df.head(2000).copy()

sample_df["keywords"] = sample_df["question"].apply(extract_market_keywords)

print(sample_df[["question", "keywords"]].head(10))

all_keywords = [
    kw
    for keyword_list in sample_df["keywords"]
    for kw in keyword_list
]

keyword_counts = Counter(all_keywords)

for keyword, count in keyword_counts.most_common(20):
    print(keyword, count)
