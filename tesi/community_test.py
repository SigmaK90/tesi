from __future__ import annotations
# QUARTO
import os
import community as community_louvain  # pip install python-louvain networkx
import networkx as nx
import pandas as pd

# File di input generati nei passaggi precedenti
NODES_INPUT = "data/polymarket_nodes_for_thesis.csv"
EDGES_INPUT = "data/polymarket_edges_knn.csv"

# File di output finale (Nodi + ID Cluster)
NODES_OUTPUT = "data/polymarket_nodes_classified.csv"


def detect_communities() -> None:
    print("=== PASSO 4: COMMUNITY DETECTION VIA ALGORITMO DI LOUVAIN ===")

    # Controlli di integrità
    if not os.path.exists(NODES_INPUT):
        print(f"[ERRORE] File dei nodi '{NODES_INPUT}' non trovato.")
        return
    if not os.path.exists(EDGES_INPUT):
        print(f"[ERRORE] File degli archi '{EDGES_INPUT}' non trovato.")
        return

    print("[1/3] Caricamento dei dati e costruzione del grafo in NetworkX...")
    df_nodes = pd.read_csv(NODES_INPUT)
    df_edges = pd.read_csv(EDGES_INPUT)

    # Inizializziamo un grafo pesato non orientato
    G = nx.Graph()

    # Aggiungiamo gli archi con il loro peso semantico (la Cosine Similarity calcolata nel 5-NN)
    # L'algoritmo di Louvain userà i pesi per capire quanto sono forti i legami
    edges_tuples = [
        (str(row.source), str(row.target), float(row.weight))
        for row in df_edges.itertuples()
    ]
    G.add_weighted_edges_from(edges_tuples)

    print(
        f"      Grafo caricato con successo: {G.number_of_nodes()} nodi e {G.number_of_edges()} archi."
    )

    print("[2/3] Esecuzione dell'algoritmo di Louvain (Ottimizzazione Modularità)...")
    start_time = pd.Timestamp.now()

    # Eseguiamo la partizione ottimizzando la modularità
    # random_state=42 garantisce che i cluster siano riproducibili ogni volta che lanci il codice
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)

    end_time = pd.Timestamp.now()
    print(
        f"      Modularità ottimizzata in {(end_time - start_time).total_seconds():.2f} secondi."
    )

    # Calcoliamo la metrica di Modularità globale per la tesi
    modularity_score = community_louvain.modularity(partition, G, weight="weight")

    # Trasformiamo il dizionario della partizione {node_id: cluster_id} in un DataFrame
    df_partition = pd.DataFrame(
        list(partition.items()), columns=["id", "cluster"]
    )

    # Assicuriamoci che l'ID sia trattato come stringa per il merge
    df_nodes["id"] = df_nodes["id"].astype(str)
    df_partition["id"] = df_partition["id"].astype(str)

    print("[3/3] Merge dei cluster con i metadati originali e salvataggio...")
    # Uniamo la colonna dei cluster al file dei nodi originale
    df_final = pd.merge(df_nodes, df_partition, on="id", how="left")

    # Gestione di eventuali nodi isolati che non hanno ricevuto un cluster
    df_final["cluster"] = df_final["cluster"].fillna(-1).astype(int)

    df_final.to_csv(NODES_OUTPUT, index=False, encoding="utf-8")

    # Statistiche finali da sparare in riunione
    num_clusters = df_final["cluster"].nunique()
    print("\n========================================================")
    print(" PASSO 4 COMPLETATO CON SUCCESSO")
    print(f" File Nodi Classificato generato: {os.path.abspath(NODES_OUTPUT)}")
    print(f" Score di Modularità della rete: {modularity_score:.4f}")
    print(f" Numero totale di macro-comunità rilevate: {num_clusters}")
    print("\n Top 10 Cluster per dimensione:")
    print(df_final["cluster"].value_counts().head(10))
    print("========================================================")



detect_communities()