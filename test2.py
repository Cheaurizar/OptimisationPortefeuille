
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# Reupload requis du fichier CSV
file_path = "standardized_residuals.csv"

# Tenter de charger le fichier si disponible
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError("Veuillez réuploader le fichier 'standardized_residuals.csv'.")

# Actifs à analyser (en ignorant la colonne 'Date' si elle existe)
assets = [col for col in df.columns if col.lower() != 'date']

# Paramètres
quantile_level = 0.90
alpha = 0.99

# Liste de résultats
results = []

for asset in assets:
    residuals = df[asset].dropna()
    if residuals.empty:
        continue

    u = np.quantile(residuals, quantile_level)
    excesses = residuals[residuals > u] - u
    n = len(residuals)
    nu = len(excesses)

    if nu < 5:
        xi, beta, VaR_alpha, ES_alpha = [np.nan] * 4
    else:
        xi, loc, beta = genpareto.fit(excesses, floc=0)
        VaR_alpha = u + (beta / xi) * (((n / nu) * (1 - alpha)) ** (-xi) - 1)
        ES_alpha = (VaR_alpha + beta - xi * u) / (1 - xi) if xi < 1 else np.nan

        # Tracer la fonction de survie empirique vs. modèle GPD
        sorted_excesses = np.sort(excesses)
        empirical_sf = 1 - np.arange(1, len(excesses) + 1) / (len(excesses) + 1)
        gpd_sf = genpareto.sf(sorted_excesses, xi, loc=0, scale=beta)

        plt.figure(figsize=(8, 5))
        plt.plot(sorted_excesses, empirical_sf, marker='o', linestyle='none', label="Empirique")
        plt.plot(sorted_excesses, gpd_sf, color='red', label="GPD ajustée")
        plt.yscale("log")
        plt.xlabel("Excès (x - u)")
        plt.ylabel("Fonction de survie (log-scale)")
        plt.title(f"Survie empirique vs. GPD - {asset}")
        plt.legend()
        plt.grid(True)
        plt.show()

    results.append({
        'asset': asset,
        'threshold_95%': u,
        'xi': xi,
        'beta': beta,
        'VaR_99%': VaR_alpha,
        'ES_99%': ES_alpha,
        'num_excesses': nu
    })

# Créer un DataFrame résumé
summary_df = pd.DataFrame(results)

# Sauvegarder le résumé dans un fichier CSV
summary_path = "gpd_summary.csv"
summary_df.to_csv(summary_path, index=False)


