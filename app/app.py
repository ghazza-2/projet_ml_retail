"""
app/app.py
==========
Application Flask — Prédiction de Churn Client Retail
Exécution : python app/app.py  (depuis la racine du projet)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(BASE_DIR, '..')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
sys.path.append(ROOT_DIR)

app = Flask(__name__, template_folder='templates', static_folder='static')

# ── Chargement des artefacts ML ──────────────────────────────────────────────
model    = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.joblib'))

# Colonnes depuis X_train — source de vérité unique
X_train_ref           = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train_test', 'X_train.csv'), nrows=0)
EXPECTED_COLS         = X_train_ref.columns.tolist()
NUMERIC_COLS_TO_SCALE = scaler.feature_names_in_.tolist()

# Valeurs d'encodage
impute_values = encoders['impute_values']
country_map   = encoders['country_map']

print(f"EXPECTED_COLS : {len(EXPECTED_COLS)} colonnes")
ORDINAL_MAPPINGS = {
    'AgeCategory':     {'Inconnu': 0, '18-24': 1, '25-34': 2, '35-44': 3,
                        '45-54': 4, '55-64': 5, '65+': 6},
    'SpendingCategory':{'Low': 0, 'Medium': 1, 'High': 2, 'VIP': 3},
    'LoyaltyLevel':    {'Inconnu': 0, 'Nouveau': 1, 'Jeune': 2, 'Établi': 3, 'Ancien': 4},
    'BasketSizeCategory': {'Inconnu': 0, 'Petit': 1, 'Moyen': 2, 'Grand': 3},
}
print("✓ Modèles chargés")


# ── Fonction de prétraitement d'un client ────────────────────────────────────

def preprocess_client(form_data: dict) -> pd.DataFrame:
    """
    Applique exactement la même chaîne de transformations que preprocessing.py
    sur les données d'un seul client soumis via le formulaire.
    """
    d = {}

    # ── Numériques bruts ──────────────────────────────────────────────────────
    numeric_fields = [
        'Frequency', 'MonetaryTotal', 'MonetaryAvg', 'MonetaryStd',
        'TotalQuantity', 'AvgQuantityPerTransaction', 'FirstPurchaseDaysAgo',
        'WeekendPurchaseRatio', 'AvgDaysBetweenPurchases', 'UniqueProducts',
        'AvgProductsPerTransaction', 'NegativeQuantityCount', 'ZeroPriceCount',
        'ReturnRatio', 'TotalTransactions', 'Age', 'SupportTicketsCount',
        'SatisfactionScore', 'DaysSinceRegistration', 'RegMonth',
        'PreferredDayOfWeek', 'PreferredHour', 'UniqueCountries', 'IsPrivateIP',
    ]
    for field in numeric_fields:
        val = form_data.get(field, '')
        try:
            d[field] = float(val)
        except (ValueError, TypeError):
            d[field] = impute_values.get(field, 0)

    # ── Features engineered ───────────────────────────────────────────────────
    freq = max(d['Frequency'], 1)
    d['AvgBasketValue']         = d['MonetaryTotal'] / freq
    d['ReturnToFrequencyRatio'] = d['NegativeQuantityCount'] / (freq + 1)
    d['IPFirstOctet']           = impute_values.get('IPFirstOctet', 70)

    # ── Imputation valeurs manquantes (médiane train) ─────────────────────────
    for col, median_val in impute_values.items():
        if col in d and (d[col] != d[col] or d[col] is None):  # NaN check
            d[col] = median_val

    # ── Target Encoding — Country ─────────────────────────────────────────────
    country  = form_data.get('Country', 'United Kingdom')
    global_mean = np.mean(list(country_map.values()))
    d['Country_TargetEnc'] = country_map.get(country, global_mean)

    # ── Encodage ordinal ──────────────────────────────────────────────────────
    for col, mapping in ORDINAL_MAPPINGS.items():
        val = form_data.get(col, 'Inconnu')
        d[col] = mapping.get(val, 0)

    # ── One-Hot Encoding ──────────────────────────────────────────────────────
    # PreferredTimeOfDay (référence supprimée = 'Après-midi')
    tod = form_data.get('PreferredTimeOfDay', 'Matin')
    d['PreferredTimeOfDay_Matin'] = int(tod == 'Matin')
    d['PreferredTimeOfDay_Midi']  = int(tod == 'Midi')
    d['PreferredTimeOfDay_Soir']  = int(tod == 'Soir')

    # Region (référence = 'Afrique')
    region = form_data.get('Region', 'UK')
    for r in ['Amérique du Nord', 'Amérique du Sud', 'Asie', 'Autre',
              'Europe centrale', 'Europe continentale', "Europe de l'Est",
              'Europe du Nord', 'Europe du Sud', 'Moyen-Orient', 'Océanie', 'UK']:
        d[f'Region_{r}'] = int(region == r)

    # WeekendPreference (référence = 'Inconnu')
    wp = form_data.get('WeekendPreference', 'Inconnu')
    d['WeekendPreference_Semaine'] = int(wp == 'Semaine')
    d['WeekendPreference_Weekend'] = int(wp == 'Weekend')

    # ProductDiversity (référence = 'Explorateur')
    pd_val = form_data.get('ProductDiversity', 'Explorateur')
    d['ProductDiversity_Modéré']    = int(pd_val == 'Modéré')
    d['ProductDiversity_Spécialisé'] = int(pd_val == 'Spécialisé')

    # Gender (référence = 'F')
    gender = form_data.get('Gender', 'Unknown')
    d['Gender_M']       = int(gender == 'M')
    d['Gender_Unknown'] = int(gender == 'Unknown')

    # AccountStatus (référence = 'Active')
    status = form_data.get('AccountStatus', 'Active')
    d['AccountStatus_Closed']    = int(status == 'Closed')
    d['AccountStatus_Pending']   = int(status == 'Pending')
    d['AccountStatus_Suspended'] = int(status == 'Suspended')

    # ── Construire le DataFrame dans l'ordre exact ────────────────────────────
    df = pd.DataFrame([d])
    df = df.reindex(columns=EXPECTED_COLS, fill_value=0)

    # ── Normalisation (StandardScaler — même que preprocessing) ──────────────
    cols_scale = [c for c in NUMERIC_COLS_TO_SCALE if c in df.columns]
    df[cols_scale] = scaler.transform(df[cols_scale])

    return df


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        X = preprocess_client(form_data)

        proba     =float(model.predict_proba(X)[0][1])
        churn     = int(proba >= 0.5)
        risk_pct  = round(proba * 100, 1)

        if risk_pct < 30:
            risk_level = 'Faible'
            risk_color = 'success'
            advice     = 'Ce client est fidèle. Maintenez la relation avec des offres de fidélité.'
        elif risk_pct < 60:
            risk_level = 'Modéré'
            risk_color = 'warning'
            advice     = 'Risque modéré. Proposez une offre personnalisée ou un contact proactif.'
        else:
            risk_level = 'Élevé'
            risk_color = 'danger'
            advice     = 'Risque critique. Intervention urgente recommandée : remise, appel, enquête satisfaction.'

        return jsonify({
            'success':    True,
            'churn':      churn,
            'proba':      risk_pct,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'advice':     advice,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats')
def stats():
    """Retourne des statistiques globales pour le dashboard."""
    try:
        X_train = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train_test', 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train_test', 'y_train.csv')).squeeze()
        y_test  = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train_test', 'y_test.csv')).squeeze()

        total = len(y_train) + len(y_test)
        churned = int((y_train.sum() + y_test.sum()))

        return jsonify({
            'total_clients': total,
            'churn_count':   churned,
            'churn_rate':    round(churned / total * 100, 1),
            'features':      X_train.shape[1],
            'model_name':    type(model).__name__,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)