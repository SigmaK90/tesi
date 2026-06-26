import os
import pandas as pd

FILE_SUMMARY = "data/polymarket_cluster_keywords_summary.csv"

if not os.path.exists(FILE_SUMMARY):
    print(f"[ERRORE] Manca il file {FILE_SUMMARY}. Lancia prima lo script MMR!")
    exit()

# Carichiamo i dati reali generati dalla pipeline
df = pd.read_csv(FILE_SUMMARY)

# Ordiniamo per dimensione decrescente (dal più grande al più piccolo)
df_sorted = df.sort_values(by="cluster_size", ascending=False).reset_index(drop=True)

print("\n==================================================================")
print("             MAPPA REALE DEI CLUSTER (ORDINATI PER SIZE)          ")
print("==================================================================")
print(f" Trovati {len(df_sorted)} cluster totali con almeno 5 mercati.\n")

# Stampiamo i primi 20 per darti subito una panoramica dei pesi massimi
print(f"{'POSIZIONE':<10} | {'CLUSTER ID':<12} | {'NUMERO MERCATI (SIZE)':<22}")
print("-" * 55)
for idx, row in df_sorted.head(20).iterrows():
    print(f"Pos #{idx+1:<5} | ID: {int(row.cluster_id):<8} | Size: {int(row.cluster_size):<10} mercati")

print("-" * 55)
print("... i restanti cluster sono visualizzabili tramite ricerca rapida qui sotto.")
print("==================================================================")

# Loop interattivo per esplorare le keyword esatte senza tirare a indovinare
while True:
    print("\n--> Vuoi vedere le keyword esatte di un cluster specifico?")
    scelta = input("Inserisci la POSIZIONE (es. 1 per il più grande) o 'q' per uscire: ").strip()
    
    if scelta.lower() == 'q':
        print("Esplorazione terminata. Adesso hai i dati reali sotto controllo, bro!")
        break
        
    try:
        pos = int(scelta)
        if 1 <= pos <= len(df_sorted):
            row = df_sorted.iloc[pos - 1]
            print("\n" + "="*70)
            print(f"Dati reali per il Cluster posizionato al numero #{pos}:")
            print(f"• ID Cluster Originale: {int(row.cluster_id)}")
            print(f"• Dimensione della Comunità: {int(row.cluster_size)} mercati")
            print(f"• Keyword MMR Estratte: {row.keywords}")
            print("="*70)
        else:
            print(f"[ERRORE] Inserisci un numero compreso tra 1 e {len(df_sorted)}.")
    except ValueError:
        print("[ERRORE] Input non valido. Inserisci un numero o 'q'.")