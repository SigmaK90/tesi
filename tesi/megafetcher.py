from __future__ import annotations
# PRIMO
import os
import time
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import requests

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
OUTPUT_FILE = "data/polymarket_nodes_for_thesis.csv"
PAGE_LIMIT = 100
TIMEOUT_SECONDS = 30


def safe_float(value: Any) -> Optional[float]:
    """Converte in float in modo sicuro per evitare crash su valori nulli o malformati."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_market(
    raw_market: Dict[str, Any], event_title: str
) -> Dict[str, Any]:
    """Normalizza la struttura del mercato atomico applicando la nomenclatura formale."""
    return {
        "id": str(raw_market.get("id") or ""),
        "question": str(raw_market.get("question") or "").strip(),
        "description": str(raw_market.get("description") or "").strip(),
        "conditionId": str(raw_market.get("conditionId") or "").strip(),
        "slug": str(raw_market.get("slug") or "").strip(),
        "liquidity": safe_float(raw_market.get("liquidity")),
        "volume": safe_float(raw_market.get("volume")),
        "active": bool(raw_market.get("active", False)),
        "closed": bool(raw_market.get("closed", False)),
        "created_at": str(raw_market.get("createdAt") or "").strip(),
        "end_date": str(raw_market.get("endDate") or "").strip(),
        "event_title": event_title,  # <--- Cambiato da macro_event_title a event_title
    }

def execute_primary_extraction_pipeline() -> None:
    """Pipeline principale di estrazione basata sull'albero degli Eventi Globali."""
    headers: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    all_discovered_markets: List[Dict[str, Any]] = []
    seen_market_ids: Set[str] = set()

    # Keyword strategiche per coprire i gap geopolitici, macroeconomici e sportivi storici
    search_keywords: List[str] = [
        "hormuz",
        "world cup",
        "war",
        "middle east",
        "taiwan",
        "election",
        "russia",
        "china",
        "fed",
        "inflation",
        "olympics",
    ]

    print("=== PIPELINE PRINCIPALE DI ESTRAZIONE NODI (TOPOLOGIA FULL) ===")

    with requests.Session() as session:
        session.headers.update(headers)

        # ------------------------------------------------------------------
        # FASE 1: SCANNERIZZAZIONE DIRETTA PER VOLUME DECRESCENTE
        # ------------------------------------------------------------------
        print("\n[FASE 1] Estrazione sistematica per volumi finanziari...")
        offset = 0
        max_volume_offset = 2000  # Soglia di campionamento per i macro eventi

        while offset < max_volume_offset:
            params: Dict[str, Any] = {
                "limit": PAGE_LIMIT,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }
            try:
                response = session.get(
                    GAMMA_EVENTS_URL, params=params, timeout=TIMEOUT_SECONDS
                )

                if response.status_code == 429:
                    print("Rate limit rilevato. Pausa di 10 secondi...")
                    time.sleep(10)
                    continue

                response.raise_for_status()
                events_batch = response.json()

                if not events_batch:
                    break

                for event in events_batch:
                    event_title = str(event.get("title") or "").strip()
                    raw_markets = event.get("markets") or []

                    if isinstance(raw_markets, list):
                        for market_data in raw_markets:
                            market_id = str(market_data.get("id") or "")
                            if (
                                market_id
                                and market_id not in seen_market_ids
                            ):
                                seen_market_ids.add(market_id)
                                normalized = normalize_market(
                                    market_data, event_title
                                )
                                all_discovered_markets.append(normalized)

                print(
                    f"   Offset {offset}/{max_volume_offset} completato. Record unici stabili: {len(all_discovered_markets)}"
                )
                offset += PAGE_LIMIT
                time.sleep(0.3)

            except Exception as e:
                print(
                    f"[ERRORE NON FATALE] Interruzione all'offset {offset}: {e}"
                )
                break

        # ------------------------------------------------------------------
        # FASE 2: SCANNERIZZAZIONE MIRATA SEMANTICA VIA KEYWORDS
        # ------------------------------------------------------------------
        print("\n[FASE 2] Iniezione keyword per il recupero dei cluster geopolitici...")
        for keyword in search_keywords:
            print(f"   Querying keyword: '{keyword}'...")
            params = {"limit": PAGE_LIMIT, "search": keyword}
            try:
                response = session.get(
                    GAMMA_EVENTS_URL, params=params, timeout=TIMEOUT_SECONDS
                )
                response.raise_for_status()
                events_batch = response.json()

                keyword_added_count = 0
                for event in events_batch:
                    event_title = str(event.get("title") or "").strip()
                    raw_markets = event.get("markets") or []

                    if isinstance(raw_markets, list):
                        for market_data in raw_markets:
                            market_id = str(market_data.get("id") or "")
                            if (
                                market_id
                                and market_id not in seen_market_ids
                            ):
                                seen_market_ids.add(market_id)
                                normalized = normalize_market(
                                    market_data, event_title
                                )
                                all_discovered_markets.append(normalized)
                                keyword_added_count += 1

                print(
                    f"   -> Keyword '{keyword}' ha iniettato {keyword_added_count} nuovi nodi unici."
                )
                time.sleep(0.5)

            except Exception as e:
                print(
                    f"[ERRORE NON FATALE] Impossibile scansionare la keyword '{keyword}': {e}"
                )

    # ------------------------------------------------------------------
    # PERSISTENZA DEI DATI E SCRITTURA FINALE SU FILE CSV
    # ------------------------------------------------------------------
    print("\n[FASE 3] Scrittura e serializzazione del dataset...")
    if all_discovered_markets:
        df = pd.DataFrame(all_discovered_markets)

        # Controllo di integrità finale
        df.drop_duplicates(subset=["id"], keep="first", inplace=True)

        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print("========================================================")
        print(" PIPELINE COMPLETATA CON SUCCESSO")
        print(f" File generato: {os.path.abspath(OUTPUT_FILE)}")
        print(f" Nodi unici reali estratti per la tesi: {len(df)}")
        print("========================================================")
    else:
        print("[ERRORE CRITICO] Nessun record estratto dalla pipeline.")

execute_primary_extraction_pipeline()