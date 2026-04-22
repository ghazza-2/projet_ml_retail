# Analyse Comportementale Clientèle Retail
### Atelier Machine Learning — GI2 | 2025-2026
 
Projet complet de data science sur un dataset e-commerce de cadeaux.  
Objectif : prédire le churn client et segmenter la clientèle via une chaîne complète de traitement ML.
 
---
 
## Contexte
 
Une entreprise e-commerce de cadeaux souhaite anticiper le départ de ses clients (churn) pour mettre en place des stratégies de rétention ciblées. Un client est considéré comme churné si son dernier achat date de plus de **91 jours**.
 
- **4 372 clients** | **52 features** | **33% de churners**
- Pipeline complet : exploration → preprocessing → modélisation → déploiement Flask
---
 
## Structure du projet
 
```
Projet_ML/
├── data/
│   ├── raw/                  # Données brutes originales
│   ├── processed/            # Données nettoyées
│   └── train_test/           # X_train, X_test, y_train, y_test
├── notebooks/
│   └── exploration.py        # Analyse exploratoire des données
├── src/
│   ├── utils.py              # Fonctions utilitaires réutilisables
│   ├── preprocessing.py      # Pipeline de nettoyage et préparation
│   ├── train_model.py        # Entraînement et évaluation des modèles
│   └── predict.py            # Prédiction sur nouveaux clients
├── models/                   # Modèles et artefacts sauvegardés (.joblib)
├── app/
│   ├── app.py                # Application Flask
│   ├── templates/
│   │   ├── index.html        # Formulaire de saisie client
│   │   └── result.html       # Affichage de la prédiction
│   └── static/
│       └── style.css         # Feuille de style
├── reports/                  # Visualisations générées
├── requirements.txt          # Dépendances Python
├── .gitignore
└── README.md
```
 
---
 
## Installation
 
### 1. Cloner le dépôt
 
```bash
git clone https://github.com/<username>/Projet_ML.git
cd Projet_ML
```
 
### 2. Créer et activer l'environnement virtuel
 
```bash
# Création
python -m venv venv
 
# Activation Windows
venv\Scripts\activate
 
# Activation Mac/Linux
source venv/bin/activate
```
 
### 3. Installer les dépendances
 
```bash
pip install -r requirements.txt
```
 
---
 
## Utilisation
 
Les scripts doivent être lancés depuis la **racine du projet** dans l'ordre suivant :
 
### Étape 1 — Exploration
 
```bash
python notebooks/exploration.py
```
 
Génère les rapports dans `reports/` : distributions, heatmap de corrélation, analyse du churn.
 
### Étape 2 — Preprocessing
 
```bash
python src/preprocessing.py
```
 
Nettoie les données, applique le pipeline anti-leakage, génère les CSV dans `data/train_test/` et les artefacts dans `models/`.
 
### Étape 3 — Entraînement
 
```bash
python src/train_model.py
```
 
Compare Logistic Regression, Random Forest et XGBoost via GridSearchCV. Sauvegarde le meilleur modèle dans `models/best_model.joblib`.
 
### Étape 4 — Prédiction
 
```bash
python src/predict.py
```
 
Teste le modèle sur X_test et prédit le churn d'un client exemple.
 
### Étape 5 — Application Flask
 
```bash
python app/app.py
```
 
Lance l'interface web sur `http://127.0.0.1:5000`.
 
---
 
## Résultats
 
| Modèle | AUC-ROC CV | AUC-ROC Test | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.972 | 0.972 | 91% |
| Random Forest | 0.997 | 0.997 | 98% |
| **XGBoost** | **0.999** | **0.9998** | **99%** |
 
**Meilleur modèle : XGBoost**
 
### Clustering K-Means (k=4)
 
| Cluster | Effectif | Proportion |
|---|---|---|
| Cluster 0 | 984 | 28.1% |
| Cluster 1 | 848 | 24.2% |
| Cluster 2 | 909 | 26.0% |
| Cluster 3 | 756 | 21.6% |
 
---
 
## Pipeline anti-leakage
 
9 features leaky ont été identifiées et supprimées avant l'entraînement :
 
| Feature | Raison |
|---|---|
| `Recency` | Définition exacte du churn (Recency ≥ 91j) |
| `ChurnRiskCategory` | Construite à partir du churn (r=0.88) |
| `RFMSegment` | Champions=0% / Dormants=100% churn |
| `CustomerType` | "Perdu" = churné en français |
| `CustomerTenureDays` | 54% des churned ont tenure=0 |
| `FavoriteSeason` | Proxy de Recency |
| `PreferredMonth` | Proxy de Recency |
| `TenureRatio` | Dérivé de Recency |
| `MonetaryPerDay` | Dérivé de Recency |
 
---
 
## Application Flask
 
L'application permet de prédire le risque de churn d'un client en temps réel :
 
- Saisie des données client via un formulaire web
- Calcul automatique des features engineered
- Prédiction via le modèle XGBoost sauvegardé
- Affichage de la probabilité, du niveau de risque et d'une recommandation marketing
**Niveaux de risque :**
 
| Probabilité | Niveau | Action |
|---|---|---|
| < 25% | Faible | Programme de récompenses |
| 25–50% | Moyen | Offre de fidélisation préventive |
| 50–75% | Élevé | Coupon ou contact direct |
| > 75% | Critique | Intervention personnalisée immédiate |
 
---
 
## Dépendances principales
 
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
flask
joblib
```
 
---
