
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# Charger les résidus standardisés
df = pd.read_csv("standardized_residuals.csv")

# Choisir un actif (par exemple le S&P 500)
asset = "^GSPC"
residuals = df[asset].dropna()

# Paramètre : quantile choisi (par exemple 95 %)
quantile = 0.95
threshold = np.quantile(residuals, quantile)

# Extraire les excès : x - u pour x > u
excesses = residuals[residuals > threshold] - threshold

# Estimation des paramètres de la GPD (form: ξ, loc, scale)
# loc = 0 car on travaille avec les excès x - u
shape, loc, scale = genpareto.fit(excesses, floc=0)

print(f"Seuil (u) : {threshold:.4f}")
print(f"Paramètre de forme ξ : {shape:.4f}")
print(f"Paramètre d'échelle β : {scale:.4f}")

# Tracer la fonction de survie empirique vs. modèle GPD
sorted_excesses = np.sort(excesses)
empirical_sf = 1 - np.arange(1, len(excesses)+1) / (len(excesses)+1)
gpd_sf = genpareto.sf(sorted_excesses, shape, loc=0, scale=scale)

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


# 6. Calcul de la VaR et de l’ES pour un niveau α
alpha = 0.99
n = len(residuals)
nu = len(excesses)
# VaR à niveau α
VaR_alpha = threshold + (scale/shape) * ( ((n/nu)*(1-alpha))**(-shape) - 1 )
# ES à niveau α (si ξ < 1)
if shape < 1:
    ES_alpha = (VaR_alpha + scale - shape*threshold) / (1 - shape)
else:
    ES_alpha = np.nan

print(f"VaR à {alpha*100:.1f}% : {VaR_alpha:.4f}")
print(f"ES  à {alpha*100:.1f}% : {ES_alpha:.4f}")
