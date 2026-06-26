import os
import pandas as pd

FILE_NODI = "data/polymarket_nodes_classified.csv"
OUTPUT_FINANCE = "data/polymarket_cluster_financial_summary.csv"

if not os.path.exists(FILE_NODI):
    print(f"[ERRORE CRITICO] Manca il file '{FILE_NODI}' nella cartella corrente!")
    exit()

print("[1/3] Caricamento del dataset classificato con metriche finanziarie...")
df = pd.read_csv(FILE_NODI)

# Pulizia rapida: assicuriamoci che volume e liquidity siano numerici
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
df["liquidity"] = pd.to_numeric(df["liquidity"], errors="coerce").fillna(0.0)
df["cluster"] = df["cluster"].fillna(-1).astype(int)

print("[2/3] Aggregazione finanziaria e calcolo dei Market Share per cluster...")

# Aggreghiamo i dati per cluster
# Recuperiamo anche la prima domanda del cluster o usiamo una stringa per capire il contenuto se serve,
# ma per ora uniamo le keyword calcolate nel summary se lo abbiamo, altrimenti calcoliamo solo i dati finanziari.
cluster_finance = df.groupby("cluster").agg(
    cluster_size=("id", "count"),
    volume_totale_usd=("volume", "sum"),
    volume_medio_mercato_usd=("volume", "mean"),
    liquidita_totale_usd=("liquidity", "sum")
).reset_index()

# Calcoliamo i totali globali della piattaforma per estrarre le percentuali
volume_globale_piattaforma = cluster_finance["volume_totale_usd"].sum()
liquidita_globale_piattaforma = cluster_finance["liquidita_totale_usd"].sum()

# Calcolo del Market Share percentuale sul transato storico di Polymarket
cluster_finance["volume_share_pct"] = (cluster_finance["volume_totale_usd"] / volume_globale_piattaforma) * 100

# Se esiste il vecchio file delle keyword di RoBERTa, facciamo un merge per non perdere i nomi!
FILE_KEYWORDS = "polymarket_cluster_keywords_summary.csv"
if os.path.exists(FILE_KEYWORDS):
    df_kw = pd.read_csv(FILE_KEYWORDS)
    # Rinominiamo la colonna per il merge
    df_kw = df_kw.rename(columns={"cluster_id": "cluster"})
    cluster_finance = pd.merge(cluster_finance, df_kw[["cluster", "keywords"]], on="cluster", how="left")

# Ordiniamo i cluster dal più ricco al più povero (Volume Totale Decrescente)
df_finance_sorted = cluster_finance.sort_values(by="volume_totale_usd", ascending=False).reset_index(drop=True)

# Salviamo il report finanziario finale
df_finance_sorted.to_csv(OUTPUT_FINANCE, index=False, encoding="utf-8")

print("\n=================================================================================")
print("          🏆 CLASSIFICA REALE DEI CLUSTER PER VOLUME DI DENARO (USD) 🏆          ")
print("=================================================================================")
print(f" Volume Totale tracciato sulla piattaforma: ${volume_globale_piattaforma:,.2f}\n")

print(f"{'POS':<4} | {'CLUSTER ID':<10} | {'SIZE':<6} | {'VOLUME TOTALE (USD)':<22} | {'SHARE %':<8} | {'KEYWORDS ESTRATTE'}")
print("-" * 110)

# Stampiamo i primi 15 cluster finanziari
for idx, row in df_finance_sorted.head(15).iterrows():
    kw_str = str(row.get("keywords", "N/A"))[:45]  # Tronchiamo per non sballare il terminale
    print(f"#{idx+1:<2}  | ID: {int(row.cluster):<6} | {int(row.cluster_size):<4} | ${row.volume_totale_usd:20,.2f} | {row.volume_share_pct:6.2f}% | {kw_str}...")

print("-" * 110)
print(f"[INFO] Report finanziario completo esportato in: {os.path.abspath(OUTPUT_FINANCE)}")
print("=================================================================================")