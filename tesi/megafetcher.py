import requests
import pandas as pd
import time
import os

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

OUTPUT_FILE = "polymarket_full.csv"

def normalize_market(raw_market):
    return {
        "id": str(raw_market.get("id", "")),
        "question": str(raw_market.get("question", "")).strip(),
        "description": str(raw_market.get("description", "")).strip(),
        "conditionId": str(raw_market.get("conditionId", "")).strip(),
        "liquidity": float(raw_market.get("liquidity", 0) or 0),
        "volume": float(raw_market.get("volume", 0) or 0),
        "end_date": str(raw_market.get("endDate", "")).strip(),
    }

def fetch_and_save_all_markets():
    offset = 0
    page_limit = 500

    # se file non esiste -> scriviamo header
    write_header = not os.path.exists(OUTPUT_FILE)

    total = 0

    while True:
        try:
            print(f"Fetching offset={offset}...")

            response = requests.get(
                GAMMA_MARKETS_URL,
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": page_limit,
                    "offset": offset
                },
                timeout=60
            )

            response.raise_for_status()
            batch = response.json()

            if not batch:
                print("No more data.")
                break

            normalized = [normalize_market(m) for m in batch]
            df = pd.DataFrame(normalized)

            # append al CSV
            df.to_csv(
                OUTPUT_FILE,
                mode='a',
                header=write_header,
                index=False
            )

            write_header = False
            total += len(df)

            print(f"Saved {len(df)} rows | Total: {total}")

            if len(batch) < page_limit:
                break

            offset += page_limit

            time.sleep(1)  # anti-rate limit

        except requests.exceptions.ReadTimeout:
            print("Timeout... retrying in 5s")
            time.sleep(5)

        except Exception as e:
            print("Errore:", e)
            break

    print(f"Done. Total markets saved: {total}")

fetch_and_save_all_markets()