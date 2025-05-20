import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Liste de tickers (1 ou plusieurs)
tickers = [
    '^GSPC',    # S&P 500 (USA)
    '^GDAXI',   # DAX (Allemagne)
    '^FCHI',    # CAC 40 (France)
    '^N225',    # Nikkei 225 (Japon)
    '^BVSP',    # Bovespa (Brésil)
    '^NSEI',    # Nifty 50 (Inde)
    ]
# Période
start_date = '2016-01-01'
end_date = '2024-12-31'
# Téléchargement
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
# Cas : plusieurs tickers
if isinstance(tickers, list) and len(tickers) > 1:
    # Extraire 'Adj Close' de chaque ticker
    adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers})
else:
    # Un seul ticker → extraire la colonne directement
    adj_close = data['Adj Close'].to_frame(name=tickers[0])
# Moyenne mensuelle
monthly_mean = adj_close.resample('ME').mean()
# Sauvegarde donnée brutes
monthly_mean.to_csv('donnees_mensuelles.csv')
print("Données mensuelles enregistrées dans donnees_mensuelles_moyennes.csv")
#-----------------------------------------------
# Charger les données à partir du CSV (utile si exécution séparée)
data = pd.read_csv('donnees_mensuelles.csv', index_col=0, parse_dates=True)
#-----------------------------------------------
# === Statistiques et standardisation (normalisation) ===
# Calculer moyenne et écart-type sur l'ensemble des données par variable (colonne)
means = data.mean()
stds = data.std()
print("Moyenne et écart-type par indice:")
print(pd.DataFrame({'moyenne': means, 'écart-type': stds}))
# Normalisation (standardisation) : (x - mean) / std
normalized = (data - means) / stds
# Sauvegarde des données normalisées
normalized.to_csv('normalized_indices_monthly.csv')
print("Données normalisées enregistrées dans normalized_indices_monthly.csv")

