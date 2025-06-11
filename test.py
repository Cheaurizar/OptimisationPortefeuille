
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# Fichier source
file_path = "standardized_residuals.csv"

# Paramètres de seuil et niveau de risque
quantile_level = 0.75
alpha = 0.99

# Charger les données
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError("❌ Fichier non trouvé. Veuillez réuploader 'standardized_residuals.csv'.")

# Supprimer la colonne Date si présente
assets = [col for col in df.columns if col.lower() != 'date']
results = []

for asset in assets:
    print(f"\n--- Traitement de l’actif : {asset} ---")
    residuals = df[asset].dropna()

    # Prendre la valeur absolue des résidus
    abs_residuals = residuals.abs()

    if len(abs_residuals) < 30:
        print(f"⚠️ Trop peu de données pour l’actif {asset} → ignoré.")
        continue

    # Définir le seuil u (quantile élevé)
    u = np.quantile(abs_residuals, quantile_level)

    # Excès au-delà du seuil
    excesses = abs_residuals[abs_residuals > u] - u
    n = len(abs_residuals)
    nu = len(excesses)

    if nu < 5:
        xi, beta, VaR_alpha, ES_alpha = [np.nan] * 4
    else:
        # Ajustement GPD
        xi, loc, beta = genpareto.fit(excesses, floc=0)

        # Calcul de la VaR et de l'ES
        VaR_alpha = u + (beta / xi) * (((n / nu) * (1 - alpha)) ** (-xi) - 1)
        ES_alpha = (VaR_alpha + beta - xi * u) / (1 - xi) if xi < 1 else np.nan

        # Tracer la fonction de survie
        sorted_excesses = np.sort(excesses)
        empirical_sf = 1 - np.arange(1, len(excesses) + 1) / (len(excesses) + 1)
        gpd_sf = genpareto.sf(sorted_excesses, xi, loc=0, scale=beta)

        plt.figure(figsize=(8, 5))
        plt.plot(sorted_excesses, empirical_sf, marker='o', linestyle='none', label="Empirique")
        plt.plot(sorted_excesses, gpd_sf, color='red', label="GPD ajustée")
        plt.yscale("log")
        plt.xlabel("Excès (|x| - u)")
        plt.ylabel("Fonction de survie (log-scale)")
        plt.title(f"Survie empirique vs. GPD - {asset}")
        plt.legend()
        plt.grid(True)
        plt.show()

    results.append({
        'asset': asset,
        'threshold': u,
        'xi': xi,
        'beta': beta,
        'VaR_99%': VaR_alpha,
        'ES_99%': ES_alpha,
        'num_excesses': nu
    })

# Résumé CSV
summary_df = pd.DataFrame(results)
summary_df.to_csv("gpd_summary_abs_residuals.csv", index=False)
print("\n✅ Résultats enregistrés dans 'gpd_summary_abs_residuals.csv'")

