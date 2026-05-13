import pandas as pd
from keybert import KeyBERT
from collections import Counter
import time

# ======================
# CONFIG
# ======================
CSV_FILE = "polymarket_full.csv"
OUTPUT_FILE = "polymarket_classified_v4.csv"

SAMPLE_SIZE = 40000
TOP_N = 5
BATCH_SIZE = 100

# ======================
# HARD FILTERS
# ======================
STOPWORDS_EXTRA = {
    "win", "score", "game", "party", "price",
    "will", "would", "could", "market",
    "yes", "no"
}

BAD_KEYWORDS = {
    "win", "price", "score", "game", "party",
    "team", "player"
}

# ======================
# BETTING TERMS
# ======================
BETTING_TERMS = {
    "spread", "exact score", "total", "odd",
    "draw", "over", "under",
    "kill", "kills", "map", "slay",
    "destroy", "inhibitor", "round"
}

# ======================
# TOPICS (ENHANCED)
# ======================
TOPIC_KEYWORDS = {
    "politics": [
        "election", "senate", "house", "democratic",
        "republican", "president", "nominee", "vote"
    ],
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth",
        "solana", "xrp", "dogecoin",
        "crypto", "token", "coin",
        "market cap", "etf", "trading"
    ],
    "sports": [
        "nba", "nfl", "nhl", "mlb",
        "final", "cup", "goal", "fc",
        "match", "club", "vs"
    ],
    "esports": [
        "kill", "kills", "map", "slay",
        "inhibitor", "quadra", "baron",
        "nashor"
    ],
    "weather": [
        "temperature", "rain", "snow", "weather"
    ],
    "economics": [
        "inflation", "rate", "gdp", "economy"
    ],
    "entertainment": [
        "album", "movie", "song",
        "music", "film", "eurovision"
    ],
    "geopolitics": [
        "war", "ceasefire", "conflict",
        "china", "russia", "ukraine", "taiwan",
        "israel", "gaza", "iran",
        "military", "invasion", "sanctions",
        "nato"
    ],
    "ai": [
        "ai", "artificial intelligence",
        "ai model", "language model", "llm",
        "chatgpt", "openai",
        "claude", "anthropic",
        "gemini", "google ai",
        "grok", "xai",
        "deepmind",
        "model release", "ai system"
    ]
}

# ======================
# LOAD
# ======================
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=["question"])
df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# ======================
# MODEL
# ======================
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

# ======================
# FUNCTIONS
# ======================

def extract_keywords(text):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=TOP_N
    )
    return [kw[0].lower().strip() for kw in keywords]


def clean_keywords(keyword_list):
    cleaned = []
    for kw in keyword_list:
        kw = kw.replace("-", " ")
        kw = kw.replace(" v ", " vs ")
        kw = kw.replace(" v", " vs")

        if any(char.isdigit() for char in kw):
            continue

        if len(kw) <= 2:
            continue

        if kw in STOPWORDS_EXTRA or kw in BAD_KEYWORDS:
            continue

        if kw.endswith("s"):
            kw = kw[:-1]

        cleaned.append(kw)

    return cleaned


def classify_type(keywords):
    for kw in keywords:
        if any(term in kw for term in BETTING_TERMS):
            return "betting"
    return "informational"


def classify_topic(keywords):
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}

    for kw in keywords:
        kw_tokens = kw.split()

        for topic, terms in TOPIC_KEYWORDS.items():
            for term in terms:

                # EXACT MATCH BOOST
                if kw == term:
                    scores[topic] += 3

                # TOKEN MATCH
                elif term in kw_tokens:
                    scores[topic] += 2

                # PARTIAL MATCH
                elif term in kw:
                    scores[topic] += 1

    best_topic = max(scores, key=scores.get)

    # CONFIDENCE FILTER
    if scores[best_topic] < 2:
        return "other"

    return best_topic


# ======================
# PROCESSING
# ======================
results = []

for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE].copy()

    print(f"Processing batch {i} → {i+BATCH_SIZE}")

    batch.loc[:, "keywords_raw"] = batch["question"].apply(extract_keywords)
    batch.loc[:, "keywords_clean"] = batch["keywords_raw"].apply(clean_keywords)

    batch.loc[:, "type"] = batch["keywords_clean"].apply(classify_type)
    batch.loc[:, "topic"] = batch["keywords_clean"].apply(classify_topic)

    results.append(batch)

    time.sleep(0.1)

# ======================
# FINAL
# ======================
df_final = pd.concat(results)
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved -> {OUTPUT_FILE}")

# ======================
# STATS
# ======================
print("\n=== DATASET STATS ===")
print(df_final["type"].value_counts())

print("\n=== TOPICS ===")
print(df_final["topic"].value_counts())


# ======================
# KEYWORD ANALYSIS
# ======================
def print_top_keywords(df_subset, title):
    all_kw = [
        kw
        for kws in df_subset["keywords_clean"]
        for kw in kws
    ]

    counts = Counter(all_kw)

    print(f"\n--- {title} ---")
    for kw, count in counts.most_common(20):
        print(kw, count)


df_betting = df_final[df_final["type"] == "betting"]
df_info = df_final[df_final["type"] == "informational"]

print_top_keywords(df_betting, "TOP BETTING KEYWORDS")
print_top_keywords(df_info, "TOP INFORMATIONAL KEYWORDS")


# ======================
# CROSS ANALYSIS
# ======================
print("\n=== TYPE vs TOPIC ===")
print(pd.crosstab(df_final["type"], df_final["topic"]))