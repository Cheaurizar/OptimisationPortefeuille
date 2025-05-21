import pandas as pd

# Vérifier la présence du package arch pour GARCH
try:
    from arch import arch_model
except ImportError:
    raise ImportError("Le package 'arch' n'est pas installé. Veuillez exécuter : pip install arch")

# Charger les données depuis le CSV
df = pd.read_csv('normalized_indices_monthly.csv', index_col=0, parse_dates=True)

# Préparer un DataFrame pour les résidus standardisés
std_resid_df = pd.DataFrame(index=df.index)

# Ajustement d'un GARCH(1,1) sur chaque actif et extraction des résidus standardisés
for asset in df.columns:
    series = df[asset].dropna()
    # GARCH(1,1) avec moyenne constante
    model = arch_model(series, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    res = model.fit(disp='off')  # Estimation par maximum de vraisemblance
    std_resid_df[asset] = res.std_resid

# Sauvegarder les résidus standardisés dans un nouveau fichier CSV
output_path = 'standardized_residuals.csv'
std_resid_df.to_csv(output_path)

print(f"Résidus standardisés enregistrés dans : {output_path}")
