from __future__ import annotations
# QUINTO
import os
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# CONFIGURAZIONE E CARICAMENTO MODELLO CORRETTO
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
print(f"[1/3] Caricamento del modello RoBERTa ({MODEL_NAME}) in locale...")
model = SentenceTransformer(MODEL_NAME)

# Cerca parole (o coppie di n-grammi significative)
def _candidate_phrases(doc: str, ngram_max: int, nr_candidates: int) -> List[str]:
    vectorizer = CountVectorizer(ngram_range=(1, ngram_max), stop_words="english") # Messo english per pulizia
    matrix = vectorizer.fit_transform([doc])
    features = vectorizer.get_feature_names_out()
    counts = matrix.toarray()[0]
    ranked = sorted(
        (
            (phrase.strip(), int(count), phrase.count(" ") + 1)
            for phrase, count in zip(features, counts)
            if phrase and not phrase.strip().isdigit()
        ),
        key=lambda item: (-item[1], -item[2], -len(item[0]), item[0]),
    )
    out: List[str] = []
    seen = set()
    for phrase, _, _ in ranked:
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= nr_candidates:
            break
    return out

# Calcolo semantico: encoding del modello e prodotto tra vettori (cosine similarity)
def _semantic_scores(doc: str, phrases: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    doc_embedding = model.encode([doc], normalize_embeddings=True, convert_to_numpy=True)
    phrase_embeddings = model.encode(list(phrases), normalize_embeddings=True, convert_to_numpy=True)
    scores = np.matmul(phrase_embeddings, doc_embedding[0])
    return phrase_embeddings, scores

# Maximum Marginal Relevance: MMR = (1-diversity)xSomiglianza col Cluster-diversity x Somiglianza con le parole già scelte
def _mmr_indices(phrase_embeddings: np.ndarray, scores: np.ndarray, top_n: int, diversity: float) -> List[int]:
    if len(scores) == 0: return []
    top_n = min(top_n, len(scores))
    if top_n <= 0: return []

    selected = [int(np.argmax(scores))]
    if top_n == 1: return selected

    candidate_similarity = np.matmul(phrase_embeddings, phrase_embeddings.T)
    remaining = set(range(len(scores))) - set(selected)

    while remaining and len(selected) < top_n:
        next_idx = None
        next_score = None
        for idx in remaining:
            redundancy = max(candidate_similarity[idx, chosen] for chosen in selected)
            mmr_score = (1.0 - diversity) * float(scores[idx]) - diversity * float(redundancy)
            if next_score is None or mmr_score > next_score:
                next_idx = idx
                next_score = mmr_score
        if next_idx is None: break
        selected.append(next_idx)
        remaining.remove(next_idx)

    return selected


def extract_cluster_keywords_local(
    doc: str, 
    top_n: int = 10, 
    diversity: float = 0.6, 
    ngram_max: int = 2
) -> List[str]:
    # Funzione master che unisce i passaggi ed estrae le keyword.
    phrases = _candidate_phrases(doc, ngram_max=ngram_max, nr_candidates=100)
    if not phrases:
        return []

    phrase_embeddings, scores = _semantic_scores(doc, phrases)
    
    # Filtro di sbarramento minimo (score_threshold = 0.10 come da suo default)
    keep = [idx for idx, score in enumerate(scores) if float(score) >= 0.10]
    if not keep: return []
    
    phrases = [phrases[idx] for idx in keep]
    phrase_embeddings = phrase_embeddings[keep]
    scores = scores[keep]

    ranked_indices = _mmr_indices(phrase_embeddings, scores, top_n=top_n, diversity=diversity)
    return [phrases[idx] for idx in ranked_indices]

# EXECUTION: APPLICAZIONE SUL DATASET DI TESI
if __name__ == "__main__":
    # Il file generato dal passo precedente (Louvain)
    FILE_NODI_CLUSTER = "data/polymarket_nodes_classified.csv" 
    OUTPUT_SUMMARY_FILE = "data/polymarket_cluster_keywords_summary.csv"
    
    if not os.path.exists(FILE_NODI_CLUSTER):
        print(f"[ERRORE CRITICO] Manca il file {FILE_NODI_CLUSTER} generato dallo step di Louvain.")
    else:
        print(f"[2/3] Caricamento nodi da {FILE_NODI_CLUSTER}...")
        df_nodi = pd.read_csv(FILE_NODI_CLUSTER)
        
        if "cluster" not in df_nodi.columns:
            print("[ERRORE] Il CSV caricato non contiene la colonna 'cluster'. Verifica lo step precedente.")
            exit()
            
        # Identifichiamo tutti i cluster unici calcolati da Louvain
        cluster_ids = sorted(df_nodi["cluster"].dropna().unique())
        print(f"      Rilevati {len(cluster_ids)} cluster unici nel grafo.")
        print("[3/3] Estrazione delle keyword via RoBERTa + MMR in corso su tutti i cluster...")
        
        summary_records = []

        # Ciclo automatico su ogni singola comunità del grafo
        for cid in cluster_ids:
            df_sub = df_nodi[df_nodi["cluster"] == cid]
            
            # Sbarramento dimensionale: saltiamo i micro-cluster con meno di 5 mercati (rumore)
            if len(df_sub) < 5:
                continue
                
            print(f"  [•] Elaborazione Cluster #{cid} ({len(df_sub)} mercati)...")
            
            # Aggreghiamo i testi delle domande appartenenti a QUESTO specifico cluster
            testo_aggregato_cluster = " ".join(df_sub["question"].astype(str).tolist())
            
            # Lancio dell'algoritmo MMR puro
            keywords_estratte = extract_cluster_keywords_local(
                doc=testo_aggregato_cluster, 
                top_n=10,         # Manteniamo le 10 keyword originali richieste
                diversity=0.6,    # Bilanciamento MMR di default
                ngram_max=2       # Unigrammi e bigrammi
            )
            
            kw_string = ", ".join(keywords_estratte)
            print(f"      --> Keywords: {kw_string}")
            
            # Accumuliamo i record per l'esportazione tabellare della tesi
            summary_records.append({
                "cluster_id": cid,
                "cluster_size": len(df_sub),
                "keywords": kw_string
            })

        # Generazione del file CSV di riepilogo finale
        df_summary = pd.DataFrame(summary_records)
        df_summary.to_csv(OUTPUT_SUMMARY_FILE, index=False, encoding="utf-8")
        
        print("\n========================================================")
        print(" PIPELINE DI ETICHETTATURA COMPLETATA CON SUCCESSO!")
        print(f" Tabella riassuntiva salvata in: {os.path.abspath(OUTPUT_SUMMARY_FILE)}")
        print("========================================================")