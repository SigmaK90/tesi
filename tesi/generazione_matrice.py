from __future__ import annotations
# SECONDO
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

NODES_INPUT = "data/polymarket_nodes_for_thesis.csv"
EMBEDDINGS_OUTPUT = "data/polymarket_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"


def generate_missing_embeddings() -> None:
    print("=== PIPELINE RIPRISTINO: GENERAZIONE MATRICE EMBEDDING ===")

    if not os.path.exists(NODES_INPUT):
        print(f"[ERRORE CRITICO] Manca il file dei nodi '{NODES_INPUT}'. Generalo prima.")
        return
    # Importo il dataset dei mercati estratto da megafetcher.py
    print(f"[1/3] Caricamento nodi da {NODES_INPUT}...")
    df = pd.read_csv(NODES_INPUT)
    
    # Pulizia stringhe per evitare crash su valori nulli
    df["event_title"] = df["event_title"].fillna("").astype(str)
    df["question"] = df["question"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    # Combino in pipe evento, domanda e descrizione per avere un blocco semantico ricco di informazioni relative al mercato
    print("      Costruzione dei blocchi semantici densi...")
    combined_texts = [
        f"Event: {row.event_title} | Question: {row.question} | Context: {row.description}".strip()
        for row in df.itertuples()
    ]
    # Lancio RoBERTa
    print(f"[2/3] Inizializzazione modello {MODEL_NAME} e calcolo vettoriale...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"      Calcolo in corso su device: {model.device}...")
    start_time = pd.Timestamp.now()
    
    embeddings = model.encode(
        combined_texts,
        batch_size=128, # Non carico tutto in RAM in una volta per evitare sovraccarichi
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    end_time = pd.Timestamp.now()
    print(f"      Matrice calcolata in {(end_time - start_time).total_seconds():.2f} secondi.")
    print(f"      Dimensioni della matrice generata: {embeddings.shape}")

    print(f"[3/3] Salvataggio binario su disco: {EMBEDDINGS_OUTPUT}...")
    # Salvo la matrice di embeddings su disco
    np.save(EMBEDDINGS_OUTPUT, embeddings)
    print("=== MATRICE RIPRISTINATA CON SUCCESSO ===")



generate_missing_embeddings()