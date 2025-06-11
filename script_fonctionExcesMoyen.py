import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# ---------------------------------------------------------------------------------------------------------------------
# 2. Charger les résidus standardisés
# ---------------------------------------------------------------------------------------------------------------------
# Remplacez le chemin ci-dessous par celui de votre fichier local si nécessaire
csv_path = "standardized_residuals.csv"
df = pd.read_csv(csv_path)

# Identifier les colonnes correspondant aux actifs (on ignore la colonne 'Date' si elle existe)
assets = [col for col in df.columns if col.lower() != "date"]

# ---------------------------------------------------------------------------------------------------------------------
# 3. Fonction pour tracer la fonction d’excès moyen (mean excess plot)
# ---------------------------------------------------------------------------------------------------------------------
def mean_excess_plot(residuals, asset_name):
    """
    Affiche la fonction d'excès moyen e(u) pour une série de résidus donnée.
    L'utilisateur pourra ensuite choisir visuellement un seuil u.
    """
    # On restreint la plage des seuils entre le 91e et le 99.5e percentile pour rester dans la queue
    p_min, p_max = 91, 99.5
    u_min = np.percentile(residuals, p_min)
    u_max = np.percentile(residuals, p_max)
    
    # Générer 101 seuils uniformément espacés entre u_min et u_max
    thresholds = np.linspace(u_min, u_max, 101)
    mean_excess = []
    
    for u in thresholds:
        tail = residuals[residuals > u]
        if len(tail) > 1:
            mean_excess.append(np.mean(tail - u))
        else:
            mean_excess.append(np.nan)
    
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, mean_excess, marker='o', linestyle='-')
    plt.title(f"Mean Excess Plot pour {asset_name}")
    plt.xlabel("Seuil u")
    plt.ylabel("Excès moyen e(u)")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# 4. Boucle principale : pour chaque actif, afficher le plot, demander le seuil, ajuster la GPD et calculer VaR/ES
# ---------------------------------------------------------------------------------------------------------------------
results = []
alpha = 1.99  # Niveau pour VaR et ES

for asset in assets:
    residuals = np.abs(df[asset].dropna())  # Utilisation des valeurs absolues
    if residuals.empty:
        continue
    
    print(f"\n====================================")
    print(f"Traitement de l'actif : {asset}")
    print(f"Nombre total d'observations   : {len(residuals)}")
    
    # 3.1 Afficher le mean excess plot pour l'actif
    mean_excess_plot(residuals, asset)
    
    # 3.2 Demander à l'utilisateur de saisir manuellement le seuil u
    #     (il doit correspondre à une valeur réaliste dans les résidus)
    while True:
        try:
            u = float(input(f"Entrez le seuil (u) choisi pour {asset} : "))
            # Vérifier que le seuil est dans l'intervalle des résidus
            if u < residuals.min() or u > residuals.max():
                raise ValueError("Le seuil doit être compris entre {:.4f} et {:.4f}".format(residuals.min(), residuals.max()))
            break
        except ValueError as ve:
            print(f"Valeur invalide : {ve}. Réessayez.")
    
    # 3.3 Extraire les excès : (x - u) pour x > u
    tail = residuals[residuals > u]
    excesses = tail - u
    n = len(residuals)
    nu = len(excesses)
    print(f"Nombre d'observations > u : {nu}")
    
    # 3.4 Ajuster la loi GPD si on a suffisamment d'excès (au moins 5 points)
    if nu < 5:
        print("Pas assez d'excès pour ajuster une GPD (moins de 5).")
        xi, beta = np.nan, np.nan
        var_99, es_99 = np.nan, np.nan
    else:
        # Estimation des paramètres GPD (MLE), on force loc=0 car on travaille sur les excès
        xi, loc, beta = genpareto.fit(excesses, floc=0)
        print(f"Paramètre forme ξ  : {xi:.4f}")
        print(f"Paramètre échelle β: {beta:.4f}")
        
        # 3.5 Calcul de la VaR et ES à 99%
        #     Formules : 
        #       VaR_alpha = u + (β / ξ) * [ ((n/nu)*(1-α))^{(-ξ)} - 1 ]
        #       ES_alpha  = (VaR_alpha + β - ξ*u) / (1 - ξ)    (si ξ < 1)
        var_99 = u + (beta / xi) * (((n / nu) * (1 - alpha)) ** (-xi) - 1)
        es_99 = (var_99 + beta - xi * u) / (1 - xi) if xi < 1 else np.nan
        print(f"VaR 99% : {var_99:.4f}")
        print(f"ES  99% : {es_99:.4f}")
        
        # 3.6 Afficher la fonction de survie empirique vs la GPD ajustée
        sorted_exc = np.sort(excesses)
        empirical_sf = 1 - np.arange(1, nu + 1) / (nu + 1)
        gpd_sf = genpareto.sf(sorted_exc, xi, loc=0, scale=beta)
        
        plt.figure(figsize=(8, 5))
        plt.plot(sorted_exc, empirical_sf, 'o', label="Survie empirique")
        plt.plot(sorted_exc, gpd_sf, '-', label="Survie GPD ajustée")
        plt.yscale("log")
        plt.xlabel("Excès (x - u)")
        plt.ylabel("Fonction de survie (log-scale)")
        plt.title(f"{asset} - Survie empirique vs GPD")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # 3.7 Stocker les résultats dans la liste pour résumé final
    results.append({
        "asset": asset,
        "chosen_threshold": u,
        "num_excesses": nu,
        "xi": xi,
        "beta": beta,
        "VaR_99%": var_99,
        "ES_99%": es_99
    })

# ---------------------------------------------------------------------------------------------------------------------
# 4. Après avoir traité tous les actifs, créer un DataFrame résumé et l'enregistrer en CSV
# ---------------------------------------------------------------------------------------------------------------------
summary_df = pd.DataFrame(results)
output_csv = "gpd_summary_manual_thresholds.csv"
summary_df.to_csv(output_csv, index=False)
print(f"\nRésumé GPD sauvegardé dans le fichier : {output_csv}")
