# =============================================================================
# src/train_model.py
# Entraînement : Classification (Churn) + ACP + Clustering K-Means
#
# Corrections appliquées vs version précédente :
#   [BUG 1] ACP PC1=100% : les colonnes ordinales (PreferredHour, AgeCategory…)
#           n'étaient pas scalées → elles dominaient toute la variance.
#           FIX : StandardScaler appliqué sur TOUTES les colonnes avant ACP/KMeans.
#
#   [BUG 2] KMeans collapse (3494/1/1/1) : outliers extrêmes (jusqu'à 45σ)
#           attiraient chaque centroïde sur un point isolé.
#           FIX : clipping à ±5σ après normalisation + KMeans sur espace PCA réduit.
#
# Exécution : python src/train_model.py
# Entrée    : data/train_test/X_train.csv, X_test.csv, y_train.csv, y_test.csv
# Sorties   : models/  +  reports/
# =============================================================================

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.cluster         import KMeans
from sklearn.decomposition   import PCA
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost non installé — pip install xgboost")
    XGBOOST_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_TEST_DIR = os.path.join(BASE_DIR, '..', 'data', 'train_test')
MODELS_DIR     = os.path.join(BASE_DIR, '..', 'models')
REPORTS_DIR    = os.path.join(BASE_DIR, '..', 'reports')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Nombre de composantes PCA pour le clustering (explique ~72% de la variance)
PCA_N_COMPONENTS_CLUSTER = 15
# Clipping des outliers avant ACP/KMeans (en nb de sigma)
CLIP_SIGMA = 5
# Nombre de clusters K-Means
K_OPTIMAL = 4


# =============================================================================
# ÉTAPE 1 — CHARGEMENT
# =============================================================================

print("=" * 60)
print("ÉTAPE 1 — CHARGEMENT")
print("=" * 60)

X_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'y_train.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'y_test.csv')).squeeze()

# Supprimer les colonnes texte résiduelles (sécurité)
cols_texte = X_train.select_dtypes(include=['object']).columns.tolist()
if cols_texte:
    print(f"  Colonnes texte supprimées : {cols_texte}")
    X_train = X_train.drop(columns=cols_texte)
    X_test  = X_test.drop(columns=cols_texte)

print(f"X_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"Churn=1 dans y_train : {y_train.mean()*100:.1f}%")


# =============================================================================
# PRÉPARATION POUR ACP & CLUSTERING
# =============================================================================
# Les données X_train issues du preprocessing sont partiellement scalées :
# les colonnes continues sont centrées-réduites, MAIS les colonnes ordinales
# (PreferredHour ≈ 12, AgeCategory ≈ 2.7…) gardent leur échelle brute.
# ACP et KMeans sont sensibles à l'échelle → on rescale TOUT ici.
#
# Cette opération est distincte du scaler du preprocessing (qui sert à la
# classification). Ce scaler_viz est uniquement pour ACP et KMeans.

print("\n  Normalisation complète pour ACP/KMeans (toutes colonnes)…")
scaler_viz = StandardScaler()
X_scaled   = scaler_viz.fit_transform(X_train)          # fit sur train uniquement
X_test_scaled = scaler_viz.transform(X_test)

# Clipping des outliers extrêmes à ±CLIP_SIGMA
# Raison : des valeurs à 45σ (ex: MonetaryStd) attirent les centroïdes K-Means
# sur des points isolés → tous les autres clients dans un seul mega-cluster.
X_clipped      = np.clip(X_scaled,      -CLIP_SIGMA, CLIP_SIGMA)
X_test_clipped = np.clip(X_test_scaled, -CLIP_SIGMA, CLIP_SIGMA)

print(f"  Clipping outliers à ±{CLIP_SIGMA}σ appliqué")
print(f"  Prêt pour ACP et K-Means")


# =============================================================================
# ÉTAPE 2 — CLASSIFICATION : PRÉDICTION DU CHURN
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 2 — CLASSIFICATION (prédiction Churn)")
print("=" * 60)

# La classification utilise X_train original (déjà scalé par preprocessing.py)
# On n'utilise PAS X_clipped ici : le clipping réduirait le signal pour les modèles
# arborescents (Random Forest, XGBoost) qui gèrent naturellement les outliers.

models = {
    'LogisticRegression': {
        'model': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'params': {
            'C':      [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'params': {
            'n_estimators':      [100, 200],
            'max_depth':         [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    },
}

if XGBOOST_AVAILABLE:
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    models['XGBoost'] = {
        'model': XGBClassifier(
            scale_pos_weight=ratio,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'params': {
            'n_estimators':  [100, 200],
            'max_depth':     [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
    }

results = {}

for name, config in models.items():
    print(f"\n--- Entraînement : {name} ---")

    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best         = grid.best_estimator_
    y_pred       = best.predict(X_test)
    y_pred_proba = best.predict_proba(X_test)[:, 1]
    auc          = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        'model':        best,
        'best_params':  grid.best_params_,
        'cv_auc':       round(grid.best_score_, 4),
        'test_auc':     round(auc, 4),
        'y_pred':       y_pred,
        'y_pred_proba': y_pred_proba
    }

    print(f"  Meilleurs params : {grid.best_params_}")
    print(f"  AUC-ROC CV      : {grid.best_score_:.4f}")
    print(f"  AUC-ROC Test    : {auc:.4f}")
    print(f"\n  Rapport de classification :")
    print(classification_report(y_test, y_pred, target_names=['Fidèle', 'Parti']))

    joblib.dump(best, os.path.join(MODELS_DIR, f'{name}.joblib'))


# Meilleur modèle
best_name  = max(results, key=lambda k: results[k]['test_auc'])
best_model = results[best_name]['model']

print("\n" + "=" * 60)
print(f"MEILLEUR MODÈLE : {best_name}")
print(f"AUC-ROC Test    : {results[best_name]['test_auc']}")
print("=" * 60)

joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))
print(f"Modèle sauvegardé : models/best_model.joblib")


# ── Visualisations classification ──────────────────────────────────────────

# Matrice de confusion
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
ConfusionMatrixDisplay(cm, display_labels=['Fidèle', 'Parti']).plot(ax=ax, cmap='Blues')
ax.set_title(f'Matrice de confusion — {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'), dpi=120)
plt.close()
print(f"Matrice de confusion sauvegardée : reports/confusion_matrix.png")

# Courbes ROC
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('Taux faux positifs')
ax.set_ylabel('Taux vrais positifs')
ax.set_title('Courbes ROC — comparaison des modèles')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'roc_curves.png'), dpi=120)
plt.close()
print(f"Courbes ROC sauvegardées : reports/roc_curves.png")

# Importance des features
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(
        best_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.invert_yaxis()
    ax.set_title(f'Top 20 features importantes — {best_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), dpi=120)
    plt.close()
    print(f"Importance des features sauvegardée : reports/feature_importance.png")


# =============================================================================
# ÉTAPE 3 — ACP (visualisation 2D)
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 3 — ACP (visualisation 2D)")
print("=" * 60)

# [BUG 1 CORRIGÉ] On utilise X_clipped (toutes colonnes scalées + outliers clippés)
# au lieu de X_train brut. Sans ça, PreferredHour (mean=12) dominait PC1 à 100%.

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_clipped)

ev = pca_2d.explained_variance_ratio_
print(f"Variance expliquée — PC1 : {ev[0]*100:.1f}%  |  PC2 : {ev[1]*100:.1f}%")
print(f"Total             : {sum(ev)*100:.1f}%")
print(f"  (Normal : l'ACP 2D sur 56 features ne peut capturer qu'une fraction de la variance)")

# Visualisation colorée par Churn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gauche : coloré par Churn
sc = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                     c=y_train, cmap='coolwarm', alpha=0.4, s=10)
axes[0].set_xlabel(f'PC1 ({ev[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
axes[0].set_title('ACP 2D — coloré par Churn\n(rouge=parti, bleu=fidèle)')
plt.colorbar(sc, ax=axes[0])

# Droite : variance cumulée expliquée par les N premières composantes
pca_full = PCA(random_state=42)
pca_full.fit(X_clipped)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'b-o', markersize=3)
axes[1].axhline(95, color='red', linestyle='--', label='95% variance')
axes[1].axhline(80, color='orange', linestyle='--', label='80% variance')
axes[1].set_xlabel('Nombre de composantes')
axes[1].set_ylabel('Variance cumulée (%)')
axes[1].set_title('Variance cumulée expliquée par l\'ACP')
axes[1].legend()
axes[1].set_xlim(1, min(30, len(cumvar)))

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'pca_2d.png'), dpi=120)
plt.close()
print(f"ACP 2D sauvegardée : reports/pca_2d.png")

# Sauvegarde du scaler et de l'ACP pour usage en production (Flask)
joblib.dump(scaler_viz, os.path.join(MODELS_DIR, 'scaler_viz.joblib'))
joblib.dump(pca_2d,     os.path.join(MODELS_DIR, 'pca.joblib'))


# =============================================================================
# ÉTAPE 4 — CLUSTERING K-MEANS
# =============================================================================

print("\n" + "=" * 60)
print("ÉTAPE 4 — CLUSTERING K-MEANS")
print("=" * 60)

# [BUG 2 CORRIGÉ] K-Means sur espace PCA réduit au lieu de X_train brut.
#
# Problème original :
#   - X_train brut contient des outliers jusqu'à 45σ après normalisation
#   - Chaque outlier extrême "capturait" un centroïde → cluster de 1 client
#   - 99.9% des clients se retrouvaient dans un seul cluster
#
# Solution :
#   1. Utiliser X_clipped (outliers clippés à ±5σ)
#   2. Réduire à PCA_N_COMPONENTS_CLUSTER composantes principales (~72% variance)
#      → réduit le bruit, concentre le signal, stabilise K-Means
#   3. K-Means sur cet espace réduit

pca_cluster = PCA(n_components=PCA_N_COMPONENTS_CLUSTER, random_state=42)
X_reduced   = pca_cluster.fit_transform(X_clipped)
variance_cluster = pca_cluster.explained_variance_ratio_.sum() * 100
print(f"Réduction ACP : {X_train.shape[1]} features → {PCA_N_COMPONENTS_CLUSTER} composantes")
print(f"Variance conservée : {variance_cluster:.1f}%")

# Méthode du coude sur l'espace réduit
inertias = []
k_range  = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_reduced)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(k_range), inertias, 'bo-', markersize=6)
ax.set_xlabel('Nombre de clusters k')
ax.set_ylabel('Inertie (espace PCA réduit)')
ax.set_title(f'Méthode du coude — K-Means sur {PCA_N_COMPONENTS_CLUSTER} composantes PCA')
# Annoter le coude
for i, (k, iner) in enumerate(zip(k_range, inertias)):
    ax.annotate(f'{iner:.0f}', (k, iner), textcoords="offset points",
                xytext=(0, 8), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'kmeans_elbow.png'), dpi=120)
plt.close()
print(f"Méthode du coude sauvegardée : reports/kmeans_elbow.png")

# Entraînement avec K_OPTIMAL clusters
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans.fit(X_reduced)
labels_train = kmeans.labels_

print(f"\nK-Means entraîné avec k={K_OPTIMAL}")
print("Distribution des clusters :")
unique, counts = np.unique(labels_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u} : {c:4d} clients ({c/len(labels_train)*100:.1f}%)")

# Profil de chaque cluster (taux de churn + features moyennes)
X_train_copy = X_train.copy()
X_train_copy['Cluster'] = labels_train
X_train_copy['Churn']   = y_train.values

print("\nTaux de churn par cluster :")
churn_by_cluster = X_train_copy.groupby('Cluster')['Churn'].mean()
for c, rate in churn_by_cluster.items():
    print(f"  Cluster {c} : {rate*100:.1f}% churn")

# Visualisation : clusters dans l'espace ACP 2D (déjà calculé)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                          c=labels_train, cmap='tab10',
                          alpha=0.5, s=10)
axes[0].set_xlabel(f'PC1 ({ev[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
axes[0].set_title(f'Clusters K-Means (k={K_OPTIMAL}) — vue ACP 2D')
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# Histogramme taux de churn par cluster
colors = plt.cm.tab10(np.linspace(0, 1, K_OPTIMAL))
axes[1].bar(churn_by_cluster.index, churn_by_cluster.values * 100,
            color=colors, edgecolor='white')
axes[1].axhline(y_train.mean() * 100, color='black', linestyle='--',
                linewidth=1.5, label=f'Moyenne globale ({y_train.mean()*100:.1f}%)')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Taux de churn (%)')
axes[1].set_title('Taux de churn par cluster')
axes[1].legend()
for i, (c, rate) in enumerate(churn_by_cluster.items()):
    axes[1].text(c, rate * 100 + 0.5, f'{rate*100:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'clusters_pca.png'), dpi=120)
plt.close()
print(f"Visualisation clusters sauvegardée : reports/clusters_pca.png")

# Sauvegarde K-Means et PCA cluster
joblib.dump(kmeans,      os.path.join(MODELS_DIR, 'kmeans.joblib'))
joblib.dump(pca_cluster, os.path.join(MODELS_DIR, 'pca_cluster.joblib'))
print(f"K-Means sauvegardé : models/kmeans.joblib")


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ DE LA MODÉLISATION")
print("=" * 60)

print("\nRésultats classification :")
for name, res in results.items():
    star = " ← MEILLEUR" if name == best_name else ""
    print(f"  {name:25s} — AUC CV : {res['cv_auc']}  |  AUC Test : {res['test_auc']}{star}")

print(f"\nClustering K-Means : k={K_OPTIMAL} clusters (espace PCA {PCA_N_COMPONENTS_CLUSTER}D)")
print("Distribution :")
for u, c in zip(unique, counts):
    bar = "█" * int(c / len(labels_train) * 40)
    print(f"  Cluster {u} : {bar} {c:4d} clients ({c/len(labels_train)*100:.1f}%)")

print(f"\nFichiers sauvegardés dans models/ :")
for f in sorted(os.listdir(MODELS_DIR)):
    print(f"  {f}")

print(f"\nRapports sauvegardés dans reports/ :")
for f in ['confusion_matrix.png', 'roc_curves.png', 'feature_importance.png',
          'pca_2d.png', 'kmeans_elbow.png', 'clusters_pca.png']:
    print(f"  {f}")

print("\n→ Prochaine étape : src/predict.py  puis  app/")