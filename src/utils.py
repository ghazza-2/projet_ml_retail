# =============================================================================
# src/utils.py
# Fonctions utilitaires réutilisables pour l'exploration et la préparation
# Appelées depuis le notebook ET depuis preprocessing.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -----------------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# -----------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset depuis un fichier CSV.
    Retourne un DataFrame pandas.
    """
    df = pd.read_csv(filepath)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# -----------------------------------------------------------------------------
# 2. APERÇU GÉNÉRAL
# -----------------------------------------------------------------------------

def overview(df: pd.DataFrame) -> None:
    """
    Affiche un résumé complet du dataset :
    - shape, types, premières lignes, informations mémoire
    """
    print("=" * 60)
    print("APERÇU GÉNÉRAL DU DATASET")
    print("=" * 60)

    print(f"\nDimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    print("\n--- Types de colonnes ---")
    print(df.dtypes.value_counts())

    print("\n--- 5 premières lignes ---")
    print(df.head())

    print("\n--- Informations mémoire ---")
    df.info()


# -----------------------------------------------------------------------------
# 3. VALEURS MANQUANTES
# -----------------------------------------------------------------------------

def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le nombre et le pourcentage de valeurs manquantes par colonne.
    Retourne un DataFrame trié par taux décroissant.
    Affiche uniquement les colonnes qui ont des manquants.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    result = pd.DataFrame({
        'nb_manquants': missing_count,
        'pourcentage': missing_pct
    }).query('nb_manquants > 0').sort_values('pourcentage', ascending=False)

    print("=" * 60)
    print("VALEURS MANQUANTES")
    print("=" * 60)

    if result.empty:
        print("Aucune valeur manquante détectée.")
    else:
        print(result.to_string())
        print(f"\n{len(result)} colonne(s) avec des valeurs manquantes.")

    return result


# -----------------------------------------------------------------------------
# 4. VALEURS ABERRANTES (OUTLIERS)
# -----------------------------------------------------------------------------

def check_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Détecte les valeurs aberrantes dans les colonnes numériques spécifiées
    en utilisant la méthode IQR (interquartile range).

    Une valeur est aberrante si elle est en dehors de :
        [Q1 - 1.5*IQR  ,  Q3 + 1.5*IQR]

    Retourne un DataFrame résumant les outliers par colonne.
    """
    results = []

    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((series < lower) | (series > upper)).sum()
        results.append({
            'colonne': col,
            'Q1': round(Q1, 2),
            'Q3': round(Q3, 2),
            'borne_basse': round(lower, 2),
            'borne_haute': round(upper, 2),
            'nb_outliers': n_outliers,
            'pct_outliers': round(n_outliers / len(series) * 100, 2)
        })

    result_df = pd.DataFrame(results).sort_values('nb_outliers', ascending=False)

    print("=" * 60)
    print("VALEURS ABERRANTES (méthode IQR)")
    print("=" * 60)
    print(result_df.to_string(index=False))

    return result_df


def check_sentinel_values(df: pd.DataFrame) -> None:
    """
    Détecte les valeurs sentinelles spécifiques à ce dataset :
    - SupportTickets = -1 ou 999  (données manquantes déguisées)
    - Satisfaction    = -1 ou 99  (idem)
    Ces valeurs doivent être recodées en NaN avant tout traitement.
    """
    print("=" * 60)
    print("VALEURS SENTINELLES")
    print("=" * 60)

    checks = {
        'SupportTickets': [-1, 999],
        'Satisfaction':   [-1, 99],
    }

    for col, sentinels in checks.items():
        if col in df.columns:
            for val in sentinels:
                count = (df[col] == val).sum()
                if count > 0:
                    print(f"  {col} == {val} : {count} occurrences ({round(count/len(df)*100, 1)}%)")


# -----------------------------------------------------------------------------
# 5. STATISTIQUES DESCRIPTIVES
# -----------------------------------------------------------------------------

def describe_numeric(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Affiche les statistiques descriptives (min, max, mean, std, quartiles)
    pour les colonnes numériques.
    """
    print("=" * 60)
    print("STATISTIQUES DESCRIPTIVES — colonnes numériques")
    print("=" * 60)
    stats = df[num_cols].describe().T
    print(stats.round(2).to_string())
    return stats


def describe_categorical(df: pd.DataFrame, cat_cols: list) -> None:
    """
    Affiche la distribution des modalités pour chaque colonne catégorielle.
    """
    print("=" * 60)
    print("DISTRIBUTION — colonnes catégorielles")
    print("=" * 60)

    for col in cat_cols:
        if col not in df.columns:
            continue
        print(f"\n--- {col} ---")
        vc = df[col].value_counts(dropna=False)
        pct = df[col].value_counts(dropna=False, normalize=True).mul(100).round(1)
        print(pd.DataFrame({'count': vc, 'pct(%)': pct}).to_string())


# -----------------------------------------------------------------------------
# 6. ANALYSE DE LA VARIABLE CIBLE
# -----------------------------------------------------------------------------

def analyze_target(df: pd.DataFrame, target_col: str = 'Churn') -> None:
    """
    Analyse la distribution de la variable cible Churn :
    - Compte et pourcentage des classes
    - Avertissement si déséquilibre détecté (< 20% pour la classe minoritaire)
    - Distribution du churn par segment RFM
    """
    print("=" * 60)
    print(f"ANALYSE DE LA VARIABLE CIBLE : {target_col}")
    print("=" * 60)

    counts = df[target_col].value_counts()
    pcts   = df[target_col].value_counts(normalize=True).mul(100).round(1)

    print("\nDistribution des classes :")
    print(pd.DataFrame({'count': counts, 'pct(%)': pcts}).to_string())

    minority_pct = pcts.min()
    if minority_pct < 20:
        print(f"\n⚠️  Déséquilibre détecté : classe minoritaire = {minority_pct}%")
        print("   → Prévoir : SMOTE, class_weight='balanced', ou sous-échantillonnage")
        print("   → Métriques à utiliser : F1-score, AUC-ROC (pas seulement l'accuracy)")
    else:
        print("\n✓ Classes relativement équilibrées.")

    if 'RFMSegment' in df.columns:
        print("\nTaux de churn par segment RFM :")
        print(df.groupby('RFMSegment')[target_col].mean()
                .sort_values(ascending=False)
                .round(3)
                .to_string())


# -----------------------------------------------------------------------------
# 7. CORRÉLATIONS
# -----------------------------------------------------------------------------

def get_high_correlations(df: pd.DataFrame, num_cols: list,
                           threshold: float = 0.8) -> pd.DataFrame:
    """
    Identifie les paires de features numériques fortement corrélées
    (|corrélation| > threshold).
    Utile pour détecter la multicolinéarité avant la modélisation.
    """
    corr_matrix = df[num_cols].corr().abs()

    # Masque pour ne garder que la partie supérieure (éviter les doublons)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr = (
        upper.stack()
             .reset_index()
             .rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', 0: 'correlation'})
             .query(f'correlation > {threshold}')
             .sort_values('correlation', ascending=False)
    )

    print("=" * 60)
    print(f"PAIRES FORTEMENT CORRÉLÉES (seuil = {threshold})")
    print("=" * 60)

    if high_corr.empty:
        print("Aucune paire au-dessus du seuil.")
    else:
        print(high_corr.to_string(index=False))
        print("\n→ Pour chaque paire, conserver la feature la plus pertinente métier.")

    return high_corr


# -----------------------------------------------------------------------------
# 8. VISUALISATIONS — sauvegarde dans reports/
# -----------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, num_cols: list,
                        save_path: str = 'reports/distributions.png') -> None:
    """
    Trace les histogrammes de toutes les colonnes numériques.
    Sauvegarde l'image dans reports/.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = len(num_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        df[col].dropna().hist(ax=axes[i], bins=30, color='steelblue', edgecolor='white')
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('')

    # Masquer les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distributions des features numériques', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"Histogrammes sauvegardés : {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list,
                              save_path: str = 'reports/correlation_heatmap.png') -> None:
    """
    Génère et sauvegarde la heatmap de corrélation entre features numériques.
    Utilise la palette coolwarm centrée sur 0.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    corr = df[num_cols].corr()

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        corr,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        linecolor='white'
    )
    plt.title('Matrice de corrélation — features numériques', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Heatmap de corrélation sauvegardée : {save_path}")


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'Churn',
                              save_path: str = 'reports/churn_distribution.png') -> None:
    """
    Génère un graphique en barres de la distribution de la variable cible.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    counts = df[target_col].value_counts()
    labels = ['Fidèle (0)', 'Parti (1)']

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=['steelblue', 'tomato'], edgecolor='white')

    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11)

    ax.set_title(f'Distribution de {target_col}', fontsize=13)
    ax.set_ylabel('Nombre de clients')
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Distribution de la target sauvegardée : {save_path}")