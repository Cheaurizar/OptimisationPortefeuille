import pandas as pd
import numpy as np
from scipy.stats import genpareto

# Charger les résidus standardisés
df = pd.read_csv("standardized_residuals.csv")

# Préparer la liste des actifs (toutes les colonnes sauf 'Date')
assets = [col for col in df.columns if col != 'Date']

# Paramètres
quantile_level = 0.95
alpha = 0.99

# Initialiser la liste de résultats
results = []

for asset in assets:
    residuals = df[asset].dropna()
    if residuals.empty:
        continue
    
    # 1. Seuil basé sur le quantile
    u = np.quantile(residuals, quantile_level)
    
    # 2. Excès
    excesses = residuals[residuals > u] - u
    n = len(residuals)
    nu = len(excesses)
    
    # Vérifier assez d'excès
    if nu < 5:
        xi, beta, VaR_alpha, ES_alpha = [np.nan]*4
    else:
        # 3. Estimation GPD
        xi, loc, beta = genpareto.fit(excesses, floc=0)
        
        # 4. VaR et ES
        VaR_alpha = u + (beta/xi) * (((n/nu)*(1-alpha))**(-xi) - 1)
        ES_alpha = (VaR_alpha + beta - xi*u) / (1 - xi) if xi < 1 else np.nan
    
    # Ajouter aux résultats
    results.append({
        'asset': asset,
        'threshold_95%': u,
        'xi': xi,
        'beta': beta,
        'VaR_99%': VaR_alpha,
        'ES_99%': ES_alpha,
        'num_excesses': nu
    })

# Créer DataFrame des résultats
summary_df = pd.DataFrame(results)

# Sauvegarder dans un CSV
output_path = "gpd_summary.csv"
summary_df.to_csv(output_path, index=False)

# Afficher la table pour l'utilisateur
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Résumé GPD pour chaque actif", dataframe=summary_df)

# Fournir un lien vers le fichier CSV
print(f"[Télécharger le résumé GPD en CSV]({output_path})")
