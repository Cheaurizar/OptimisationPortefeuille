import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_returns_statistics(csv_file_path):
    """
    Calcule les statistiques descriptives des rendements quotidiens à partir d'un fichier CSV.
    
    Parameters:
    csv_file_path (str): Chemin vers le fichier CSV avec les indices boursiers
    
    Returns:
    pd.DataFrame: Tableau des statistiques formatées
    """
    
    # Lecture du fichier CSV
    # On assume que la première colonne contient les dates
    df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
    
    # Calcul des rendements quotidiens (log returns)
    returns = np.log(df / df.shift(1)).dropna()
    
    # Initialisation du dictionnaire pour stocker les résultats
    stats_dict = {}
    
    # Calcul des statistiques pour chaque actif
    for column in returns.columns:
        asset_returns = returns[column].dropna()
        
        # Calcul des statistiques
        mean_val = asset_returns.mean()
        std_val = asset_returns.std()
        skew_val = stats.skew(asset_returns)
        kurt_val = stats.kurtosis(asset_returns, fisher=True)  # Excess kurtosis
        min_val = asset_returns.min()
        max_val = asset_returns.max()
        median_val = asset_returns.median()
        
        # Formatage selon les spécifications
        stats_dict[column] = {
            'Mean': f"{mean_val:.2E}",  # Notation scientifique
            'Std': round(std_val, 4),   # 4 décimales
            'Skewness': round(skew_val, 4),  # 4 décimales
            'Kurtosis': round(kurt_val, 4),  # 4 décimales
            'Max': round(max_val, 4),   # 4 décimales
            'Min': round(min_val, 4),   # 4 décimales
            'Median': f"{median_val:.2E}"  # Notation scientifique
        }
    
    # Création du DataFrame final
    result_df = pd.DataFrame(stats_dict).T
    
    # Réorganisation des colonnes dans l'ordre souhaité
    column_order = ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Max', 'Min', 'Median']
    result_df = result_df[column_order]
    
    return result_df

def create_visual_table(stats_df, save_path=None):
    """
    Crée un tableau visuel avec matplotlib et l'affiche/sauvegarde.
    
    Parameters:
    stats_df (pd.DataFrame): DataFrame avec les statistiques
    save_path (str, optional): Chemin pour sauvegarder l'image
    """
    # Configuration de la figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Préparation des données pour le tableau
    # Conversion du DataFrame en format pour matplotlib table
    table_data = []
    
    # En-têtes
    headers = ['Asset'] + list(stats_df.columns)
    table_data.append(headers)
    
    # Données
    for asset in stats_df.index:
        row = [asset] + [str(stats_df.loc[asset, col]) for col in stats_df.columns]
        table_data.append(row)
    
    # Création du tableau
    table = ax.table(cellText=table_data[1:], 
                     colLabels=table_data[0],
                     cellLoc='center', 
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Formatage du tableau
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style de l'en-tête
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style des cellules alternées
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    # Titre
    plt.title('Statistiques Descriptives des Rendements Quotidiens', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Affichage
    plt.tight_layout()
    
    # Sauvegarde si chemin spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Tableau visuel sauvegardé dans '{save_path}'")
    
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez 'votre_fichier.csv' par le chemin vers votre fichier
    csv_file = "donnees_mensuelles.csv"
    
    try:
        # Calcul des statistiques
        results = calculate_returns_statistics(csv_file)
        
        # Affichage du tableau formaté (terminal)
        # display_formatted_table(results)
        
        # Affichage visuel - Tableau matplotlib
        create_visual_table(results, "tableau_statistiques.png")
        
        # Optionnel : sauvegarde dans un fichier CSV
        results.to_csv("statistiques_rendements.csv")
        print(f"\nLes résultats ont été sauvegardés dans 'statistiques_rendements.csv'")
        
        # Affichage du DataFrame pandas standard
        print("\nDataFrame pandas :")
        print(results)
        
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{csv_file}' n'a pas été trouvé.")
        print("Assurez-vous que le chemin vers le fichier est correct.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

# Fonction pour créer un fichier d'exemple (optionnel)
def create_sample_data():
    """
    Crée un fichier CSV d'exemple avec des données simulées.
    """
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulation de prix d'indices (marche aléatoire avec drift)
    n_days = len(dates)
    
    # Simulation des prix
    tunindex = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n_days)))
    masi = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, n_days)))
    cac40 = 100 * np.exp(np.cumsum(np.random.normal(-0.00001, 0.015, n_days)))
    sp500 = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.011, n_days)))
    
    df_sample = pd.DataFrame({
        'Date': dates,
        'Tunindex': tunindex,
        'Masi': masi,
        'CAC40': cac40,
        'S&P500': sp500
    })
    
    df_sample.set_index('Date', inplace=True)
    df_sample.to_csv('indices_boursiers_exemple.csv')
    print("Fichier d'exemple 'indices_boursiers_exemple.csv' créé avec succès!")
    
    return df_sample

# Pour créer des données d'exemple et tester le script
# create_sample_data()
# results = calculate_returns_statistics('indices_boursiers_exemple.csv')
# display_formatted_table(results)
