
import pandas as pd
from arch import arch_model

def select_best_gjr_garch_order(series, p_max=3, q_max=3, criterion='aic'):
    """
    Sélectionne les meilleurs ordres (p, q) pour un modèle GJR-GARCH(p,q)
    en utilisant AIC, BIC ou log-likelihood.
    """
    best_score = float('inf') if criterion in ['aic', 'bic'] else float('-inf')
    best_order = None
    best_model = None

    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            try:
                model = arch_model(series, mean='Constant', vol='GARCH', p=p, o=1, q=q, dist='normal')
                # Remarque : GJR-GARCH = GARCH avec o=1 (asymétrie)
                res = model.fit(disp='off')
                score = {
                    'aic': res.aic,
                    'bic': res.bic,
                    'loglikelihood': res.loglikelihood
                }[criterion]

                if ((criterion in ['aic', 'bic'] and score < best_score) or
                    (criterion == 'loglikelihood' and score > best_score)):
                    best_score = score
                    best_order = (p, 1, q)  # p, o=1, q
                    best_model = res
            except:
                continue

    return best_order, best_model

# 📥 Charger les données du fichier CSV
df = pd.read_csv("normalized_indices_monthly.csv", index_col=0, parse_dates=True)

# 📊 DataFrame pour stocker les résidus standardisés
residuals_df = pd.DataFrame(index=df.index)

# 📈 Appliquer le GJR-GARCH optimal à chaque série
results_summary = {}

for asset in df.columns:
    print(f"🔍 Traitement de l’actif : {asset}")
    series = df[asset].dropna()

    if len(series) < 50:
        print(f"⚠️ Trop peu de données pour {asset}, ignoré.")
        continue

    best_order, best_model = select_best_gjr_garch_order(series, p_max=10, q_max=10, criterion='aic')
    
    if best_model is not None:
        residuals_df.loc[series.index, asset] = best_model.std_resid
        results_summary[asset] = {
            'best_order': best_order,
            'aic': best_model.aic,
            'bic': best_model.bic,
            'loglikelihood': best_model.loglikelihood
        }
        print(f"✔️ {asset} — Meilleur GJR-GARCH{best_order} (AIC: {best_model.aic:.2f})")
    else:
        print(f"❌ Échec de modélisation pour {asset}")

# 💾 Enregistrer les résidus standardisés
residuals_df.to_csv("standardized_residuals_gjr_garch.csv")
print("✅ Résidus standardisés GJR-GARCH enregistrés dans : standardized_residuals_gjr_garch.csv")

# 📄 Résumé des meilleurs modèles
summary_df = pd.DataFrame(results_summary).T
summary_df.to_csv("gjr_garch_model_selection_summary.csv")
print("📄 Résumé des modèles GJR-GARCH enregistrés dans : gjr_garch_model_selection_summary.csv")
