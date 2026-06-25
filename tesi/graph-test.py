from __future__ import annotations

import os
from typing import List
import numpy as np
import pandas as pd

# File di input generati nei passaggi precedenti
NODES_INPUT = "polymarket_nodes_for_thesis.csv"
# Matrice .npy degli embeddings generati con RoBERTa
EMBEDDINGS_INPUT = "polymarket_embeddings.npy" 

# File di output per gli archi
EDGES_OUTPUT = "polymarket_edges_knn.csv"

# Iperparametri della topologia del grafo
K_NEIGHBORS = 5
SIMILARITY_THRESHOLD = 0.50
BATCH_SIZE = 2000  # Gestione a blocchi per efficienza computazionale

def build_knn_edges() -> None:
    print("=== PASSO 1: GENERAZIONE DELLA TOPOLOGIA DELLA RETE (EDGES) ===")

    # Controlli di integrità dei file
    if not os.path.exists(NODES_INPUT):
        print(f"[ERRORE] File dei nodi '{NODES_INPUT}' non trovato.")
        return
    if not os.path.exists(EMBEDDINGS_INPUT):
        print(f"[ERRORE] Matrice degli embedding '{EMBEDDINGS_INPUT}' non trovata.")
        return

    print("[1/3] Caricamento nodi e matrici di embedding...")
    df_nodes = pd.read_csv(NODES_INPUT)
    embeddings = np.load(EMBEDDINGS_INPUT)

    num_nodes = embeddings.shape[0]
    node_ids = df_nodes["id"].astype(str).tolist()

    if len(node_ids) != num_nodes:
        print("[ERRORE] Disallineamento tra il numero di righe nel CSV e la matrice .npy!")
        return

    # Normalizzazione geometrica dei vettori alla norma L2 unitaria.
    # Questo trucco algebrico trasforma il Prodotto Scalare direttamente in Similarità Cosina.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Prevenzione divisione per zero
    normalized_embeddings = embeddings / norms

    print(f"      Pronti al calcolo su {num_nodes} nodi.")

    # Liste di accumulo per la Edge List finale
    edges_source: List[str] = []
    edges_target: List[str] = []
    edges_weight: List[float] = []

    print(f"[2/3] Calcolo dei {K_NEIGHBORS}-NN con soglia Cosine Similarity >= {SIMILARITY_THRESHOLD}...")
    start_time = pd.Timestamp.now()

    # Loop a blocchi (Batch) per evitare l'esplosione della memoria allocata
    for i in range(0, num_nodes, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, num_nodes)
        batch = normalized_embeddings[i:end_idx]

        # Moltiplicazione di matrici: calcola la similarità tra il batch attuale e TUTTI i nodi della rete
        similarity_matrix = np.dot(batch, normalized_embeddings.T)

        for batch_local_idx in range(similarity_matrix.shape[0]):
            global_node_idx = i + batch_local_idx
            source_id = node_ids[global_node_idx]
            scores = similarity_matrix[batch_local_idx]

            # Mascheramento della self-similarity (un mercato non deve fare un arco con se stesso)
            scores[global_node_idx] = -1.0

            # Estrazione dei top K indici più alti (Algoritmo parziale O(N))
            top_k_indices = np.argpartition(scores, -K_NEIGHBORS)[-K_NEIGHBORS:]
            # Ordinamento locale dei top K estratti
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])][::-1]

            for target_idx in top_k_indices:
                score = float(scores[target_idx])
                
                # Sbarramento di sicurezza: tagliamo fuori le connessioni semantiche deboli
                if score >= SIMILARITY_THRESHOLD:
                    edges_source.append(source_id)
                    edges_target.append(node_ids[target_idx])
                    edges_weight.append(score)

        if (i // BATCH_SIZE) % 5 == 0 or end_idx == num_nodes:
            print(f"      Avanzamento: {end_idx}/{num_nodes} nodi mappati...")

    end_time = pd.Timestamp.now()
    print(f"      Topologia calcolata in {(end_time - start_time).total_seconds():.2f} secondi.")

    # 3. Creazione del DataFrame degli Archi e Persistenza su disco
    print("[3/3] Scrittura della Edge List strutturata...")
    df_edges = pd.DataFrame({
        "source": edges_source,
        "target": edges_target,
        "weight": edges_weight
    })

    df_edges.to_csv(EDGES_OUTPUT, index=False, encoding="utf-8")

    print("\n========================================================")
    print(" PASSO 1 COMPLETATO CON SUCCESSO")
    print(f" File Edge List generato: {os.path.abspath(EDGES_OUTPUT)}")
    print(f" Numero totale di archi validati inseriti nel grafo: {len(df_edges)}")
    print(f" Grado medio di connessione (Densità locale): {len(df_edges) / num_nodes:.2f}")
    print("========================================================")



build_knn_edges()