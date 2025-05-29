
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("standardized_residuals.csv")

# Choisir la série de résidus à analyser (par exemple le S&P 500)
residuals = df["^GSPC"].dropna()

# Tracer la fonction d’excès moyen
thresholds = np.linspace(np.percentile(residuals, 95), np.percentile(residuals, 99.5), 100)
mean_excess = []

for u in thresholds:
    excesses = residuals[residuals > u] - u
    if len(excesses) > 0:
        mean_excess.append(np.mean(excesses))
    else:
        mean_excess.append(np.nan)

# Affichage du graphe
plt.figure(figsize=(10, 6))
plt.plot(thresholds, mean_excess, marker='o', linestyle='-')
plt.title("Fonction d’excès moyen pour ^GSPC")
plt.xlabel("Seuil u")
plt.ylabel("Excès moyen e(u)")
plt.grid(True)
plt.show()
