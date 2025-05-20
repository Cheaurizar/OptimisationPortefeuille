import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# === Boîtes à moustaches ===
# Charger les données à partir du CSV 
data = pd.read_csv('donnees_mensuelles.csv', index_col=0, parse_dates=True)
# Tracer les boîtes à moustaches avec les points
plt.figure(figsize=(14, 8))
sns.boxplot(data=data, orient='h', showfliers=True, fliersize=3)
sns.stripplot(data=data, orient='h', color='black', size=2, jitter=0.2, alpha=0.5)
plt.title('Boîtes à moustaches des indices boursiers (valeurs mensuelles)')
plt.xlabel('Valeur de l\'indice')
plt.tight_layout()
plt.show()
