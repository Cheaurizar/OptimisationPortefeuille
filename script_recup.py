import yfinance as yf
import pandas as pd
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
fichier_brute = 'donnees_mensuelles.csv'
fichier_normalise = 'donnees_mensuelles_normalisees.csv'

# Téléchargement
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers})

# Moyenne mensuelle
monthly_mean = adj_close.resample('ME').mean()

# Sauvegarde donnée brutes
monthly_mean.to_csv(fichier_brute)
print("Données mensuelles enregistrées dans donnees_mensuelles_moyennes.csv")

# Charger les données à partir du CSV
data = pd.read_csv(fichier_brute, index_col=0, parse_dates=True)

# Calculer moyenne et écart-type sur l'ensemble des données par variable (colonne) et l'afficher
means = data.mean()
stds = data.std()
print("Moyenne et écart-type par indice:")
print(pd.DataFrame({'moyenne': means, 'écart-type': stds}))

# Normalisation (standardisation) : (x - mean) / std
normalized = (data - means) / stds

# Sauvegarde des données normalisées
normalized.to_csv(fichier_normalise)
print("Données normalisées enregistrées dans normalized_indices_monthly.csv")
