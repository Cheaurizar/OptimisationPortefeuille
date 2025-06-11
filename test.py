import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, t  # t : pour simuler des résidus t-Student

# Fichier source
file_path = "standardized_residuals.csv"

# Paramètres
quantile_level = 0.95  # seuil EVT
alpha = 0.99           # niveau de VaR / ES
simulated_sample_size = 10000
# Degrés de liberté pour t-Student (adapter selon votre modèle GARCH)
df_t = 5

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
    original_residuals = df[asset].dropna()

    if len(original_residuals) < 30:
        print(f"⚠️ Trop peu de données pour l’actif {asset} → ignoré.")
        continue

    # Partie 3 : Simulation de résidus selon une loi t
    simulated_resid = t.rvs(df_t, size=simulated_sample_size)
    abs_resid = np.abs(simulated_resid)

    # Seuil EVT basé sur un quantile élevé
    u = np.quantile(abs_resid, quantile_level)
    excesses = abs_resid[abs_resid > u] - u
    n = len(abs_resid)
    nu = len(excesses)

    if nu < 5:
        print("⚠️ Pas assez d'excès pour ajuster une GPD.")
        xi, beta, VaR_alpha, ES_alpha = [np.nan] * 4
    else:
        # Ajustement GPD
        xi, loc, beta = genpareto.fit(excesses, floc=0)

        # Calcul de VaR/ES
        VaR_alpha = u + (beta / xi) * (((n / nu) * (1 - alpha)) ** (-xi) - 1)
        ES_alpha = (VaR_alpha + beta - xi * u) / (1 - xi) if xi < 1 else np.nan

        # Courbe de survie
        sorted_excesses = np.sort(excesses)
        empirical_sf = 1 - np.arange(1, len(excesses) + 1) / (len(excesses) + 1)
        gpd_sf = genpareto.sf(sorted_excesses, xi, loc=0, scale=beta)

        plt.figure(figsize=(8, 5))
        plt.plot(sorted_excesses, empirical_sf, marker='o', linestyle='none', label="Empirique")
        plt.plot(sorted_excesses, gpd_sf, color='red', label="GPD ajustée")
        plt.yscale("log")
        plt.xlabel("Excès (|x| - u)")
        plt.ylabel("Fonction de survie (log-scale)")
        plt.title(f"Survie empirique vs. GPD - {asset} (résidus simulés)")
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
summary_df.to_csv("gpd_summary_simulated_residuals.csv", index=False)
print("\n✅ Résultats enregistrés dans 'gpd_summary_simulated_residuals.csv'")
