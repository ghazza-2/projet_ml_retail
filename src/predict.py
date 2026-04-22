# =============================================================================
# src/predict.py
# Prédiction Churn — compatible avec preprocessing.py (version anti-leakage)
#
# Artefacts utilisés (générés par preprocessing.py + train_model.py) :
#   models/best_model.joblib   — modèle entraîné (XGBoost / RF / LR)
#   models/scaler.joblib       — StandardScaler fitté sur X_train
#   models/encoders.joblib     — impute_values (médianes) + country_map
#   data/train_test/X_train.csv — source de vérité pour les colonnes
#
# Exécution : python src/predict.py
# =============================================================================

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib


# =============================================================================
# CHEMINS
# =============================================================================

MODELS_DIR     = os.path.join(BASE_DIR, '..', 'models')
TRAIN_TEST_DIR = os.path.join(BASE_DIR, '..', 'data', 'train_test')
REPORTS_DIR    = os.path.join(BASE_DIR, '..', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


# =============================================================================
# MAPPINGS (identiques à preprocessing.py)
# =============================================================================

ORDINAL_MAPPINGS = {
    'AgeCategory':     {'Inconnu': 0, '18-24': 1, '25-34': 2,
                        '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6},
    'SpendingCategory':{'Low': 0, 'Medium': 1, 'High': 2, 'VIP': 3},
    'LoyaltyLevel':    {'Inconnu': 0, 'Nouveau': 1, 'Jeune': 2,
                        'Établi': 3, 'Ancien': 4},
    'BasketSizeCategory': {'Inconnu': 0, 'Petit': 1, 'Moyen': 2, 'Grand': 3},
}

NOMINAL_COLS = [
    'PreferredTimeOfDay', 'Region', 'WeekendPreference',
    'ProductDiversity', 'Gender', 'AccountStatus',
]

# Colonnes leaky et inutiles — jamais envoyées au modèle
COLS_NEVER_USE = [
    'CustomerID', 'Churn', 'NewsletterSubscribed', 'RegistrationDate',
    'LastLoginIP', 'ChurnRiskCategory', 'RFMSegment', 'CustomerType',
    'Recency', 'CustomerTenureDays', 'FavoriteSeason', 'PreferredMonth',
    'TenureRatio', 'MonetaryPerDay',
]


# =============================================================================
# CHARGEMENT DES ARTEFACTS
# =============================================================================

def load_artifacts() -> dict:
    """
    Charge les artefacts générés par preprocessing.py et train_model.py.
    La liste des colonnes attendues est lue depuis X_train.csv
    — source de vérité unique, évite tout désynchronisme.
    """
    print("Chargement des artefacts…")

    model    = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.joblib'))

    # Colonnes exactes attendues par le modèle
    X_train_ref   = pd.read_csv(
        os.path.join(TRAIN_TEST_DIR, 'X_train.csv'), nrows=0
    )
    expected_cols = X_train_ref.columns.tolist()
    scale_cols    = scaler.feature_names_in_.tolist()

    print(f"  Modèle         : {type(model).__name__}")
    print(f"  Features       : {len(expected_cols)}")
    print(f"  Cols scalées   : {len(scale_cols)}")

    return {
        'model':         model,
        'scaler':        scaler,
        'impute_values': encoders['impute_values'],
        'country_map':   encoders['country_map'],
        'expected_cols': expected_cols,
        'scale_cols':    scale_cols,
        'global_mean':   float(np.mean(list(encoders['country_map'].values()))),
    }


# =============================================================================
# PIPELINE DE PRÉTRAITEMENT — UN CLIENT
# =============================================================================

def preprocess_client(raw: dict, art: dict) -> pd.DataFrame:
    """
    Applique exactement la même chaîne que preprocessing.py sur un client.

    Étapes :
      1. Feature engineering (AvgBasketValue, ReturnToFrequencyRatio, IP)
      2. Imputation (médianes calculées sur X_train)
      3. Encodage ordinal (mapping fixe)
      4. Target encoding Country
      5. One-Hot Encoding (aligné sur les colonnes de X_train)
      6. Alignement des colonnes sur expected_cols
      7. Normalisation StandardScaler
    """
    d = dict(raw)   # copie pour ne pas modifier l'original

    # ── 1. Feature engineering ────────────────────────────────────────────────
    freq = max(float(d.get('Frequency', 1)), 1)
    monetary = float(d.get('MonetaryTotal', 0))

    d['AvgBasketValue']         = monetary / freq
    d['ReturnToFrequencyRatio'] = float(d.get('NegativeQuantityCount', 0)) / (freq + 1)

    # IPFirstOctet depuis LastLoginIP si disponible
    ip = str(d.get('LastLoginIP', ''))
    try:
        d['IPFirstOctet'] = int(ip.split('.')[0])
    except Exception:
        d['IPFirstOctet'] = float(art['impute_values'].get('IPFirstOctet', 70))

    # DaysSinceRegistration depuis RegistrationDate si disponible
    if 'RegistrationDate' in d and d['RegistrationDate']:
        try:
            reg_date = pd.to_datetime(d['RegistrationDate'], dayfirst=True, errors='coerce')
            d['DaysSinceRegistration'] = (pd.Timestamp('2011-12-31') - reg_date).days
            d['RegMonth'] = reg_date.month
        except Exception:
            pass

    # ── 2. Imputation (médianes de X_train) ───────────────────────────────────
    for col, median_val in art['impute_values'].items():
        if col not in d or d[col] is None or (
            isinstance(d[col], float) and np.isnan(d[col])
        ):
            d[col] = float(median_val)
        else:
            try:
                d[col] = float(d[col])
            except (ValueError, TypeError):
                d[col] = float(median_val)

    # ── 3. Encodage ordinal ───────────────────────────────────────────────────
    for col, mapping in ORDINAL_MAPPINGS.items():
        val = d.get(col, 'Inconnu')
        d[col] = mapping.get(str(val), 0)

    # ── 4. Target Encoding — Country ─────────────────────────────────────────
    country = str(d.get('Country', ''))
    d['Country_TargetEnc'] = art['country_map'].get(country, art['global_mean'])

    # ── 5. One-Hot Encoding ───────────────────────────────────────────────────
    # PreferredTimeOfDay (ref = Après-midi)
    tod = str(d.get('PreferredTimeOfDay', 'Matin'))
    d['PreferredTimeOfDay_Matin'] = int(tod == 'Matin')
    d['PreferredTimeOfDay_Midi']  = int(tod == 'Midi')
    d['PreferredTimeOfDay_Soir']  = int(tod == 'Soir')

    # Region (ref = Afrique)
    region = str(d.get('Region', 'UK'))
    for r in ['Amérique du Nord', 'Amérique du Sud', 'Asie', 'Autre',
              'Europe centrale', 'Europe continentale', "Europe de l'Est",
              'Europe du Nord', 'Europe du Sud', 'Moyen-Orient', 'Océanie', 'UK']:
        d[f'Region_{r}'] = int(region == r)

    # WeekendPreference (ref = Inconnu)
    wp = str(d.get('WeekendPreference', 'Inconnu'))
    d['WeekendPreference_Semaine'] = int(wp == 'Semaine')
    d['WeekendPreference_Weekend'] = int(wp == 'Weekend')

    # ProductDiversity (ref = Explorateur)
    pd_val = str(d.get('ProductDiversity', 'Explorateur'))
    d['ProductDiversity_Modéré']     = int(pd_val == 'Modéré')
    d['ProductDiversity_Spécialisé'] = int(pd_val == 'Spécialisé')

    # Gender (ref = F)
    gender = str(d.get('Gender', 'Unknown'))
    d['Gender_M']       = int(gender == 'M')
    d['Gender_Unknown'] = int(gender == 'Unknown')

    # AccountStatus (ref = Active)
    status = str(d.get('AccountStatus', 'Active'))
    d['AccountStatus_Closed']    = int(status == 'Closed')
    d['AccountStatus_Pending']   = int(status == 'Pending')
    d['AccountStatus_Suspended'] = int(status == 'Suspended')

    # ── 6. Alignement colonnes sur expected_cols ──────────────────────────────
    df = pd.DataFrame([d])
    df = df.reindex(columns=art['expected_cols'], fill_value=0)

    # ── 7. Normalisation ──────────────────────────────────────────────────────
    cols_scale = [c for c in art['scale_cols'] if c in df.columns]
    df[cols_scale] = art['scaler'].transform(df[cols_scale])

    return df


# =============================================================================
# PRÉDICTION — UN CLIENT
# =============================================================================

def predict_client(raw: dict, art: dict) -> dict:
    """
    Retourne le risque de churn pour un client brut (dict de features).
    """
    X = preprocess_client(raw, art)

    proba = float(art['model'].predict_proba(X)[0][1])
    churn = int(proba >= 0.5)

    if proba < 0.25:
        risque = 'Faible'
        recommandation = 'Client fidèle — programme de récompenses.'
    elif proba < 0.50:
        risque = 'Moyen'
        recommandation = 'Surveiller — offre de fidélisation préventive.'
    elif proba < 0.75:
        risque = 'Élevé'
        recommandation = 'Action urgente — coupon ou contact direct.'
    else:
        risque = 'Critique'
        recommandation = 'Risque maximal — intervention personnalisée immédiate.'

    return {
        'churn':          churn,
        'probabilite':    round(proba, 4),
        'risque':         risque,
        'recommandation': recommandation,
    }


# =============================================================================
# PRÉDICTION BATCH — X_TEST
# =============================================================================

def predict_batch(X: pd.DataFrame, art: dict) -> pd.DataFrame:
    """
    Prédiction sur un batch déjà préprocessé (X_test.csv).
    Aligne les colonnes sur expected_cols avant de prédire.
    """
    # Aligner sur les colonnes attendues
    X = X.reindex(columns=art['expected_cols'], fill_value=0)

    # Supprimer colonnes texte résiduelles
    text_cols = X.select_dtypes(include=['object']).columns.tolist()
    if text_cols:
        X = X.drop(columns=text_cols)

    probas = art['model'].predict_proba(X)[:, 1].astype(float)
    preds  = (probas >= 0.5).astype(int)

    return pd.DataFrame({
        'churn_pred':  preds,
        'probabilite': probas.round(4),
        'risque': pd.cut(
            probas,
            bins=[0, 0.25, 0.50, 0.75, 1.01],
            labels=['Faible', 'Moyen', 'Élevé', 'Critique']
        ),
    })


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("CHARGEMENT DES ARTEFACTS")
    print("=" * 60)
    art = load_artifacts()

    # ── TEST 1 : batch sur X_test ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1 — PRÉDICTION BATCH SUR X_TEST")
    print("=" * 60)

    X_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'y_test.csv')).squeeze()

    results_df = predict_batch(X_test, art)
    results_df['churn_reel'] = y_test.values

    accuracy = (results_df['churn_pred'] == y_test.values).mean()

    print(f"Clients analysés : {len(results_df)}")
    print(f"Accuracy         : {accuracy*100:.1f}%")
    print("\nDistribution des niveaux de risque :")
    print(results_df['risque'].value_counts().to_string())

    output_path = os.path.join(REPORTS_DIR, 'predictions_xtest.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nPrédictions sauvegardées : {output_path}")

    # ── TEST 2 : client individuel ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2 — PRÉDICTION CLIENT INDIVIDUEL")
    print("=" * 60)

    # Client à risque élevé (inactif, peu de transactions, insatisfait)
    client_a_risque = {
        'Frequency':                 2,
        'MonetaryTotal':             150.0,
        'MonetaryAvg':               75.0,
        'AvgQuantityPerTransaction': 3.0,
        'WeekendPurchaseRatio':      0.0,
        'AvgDaysBetweenPurchases':   90.0,
        'UniqueProducts':            3,
        'AvgProductsPerTransaction': 1.5,
        'ZeroPriceCount':            0,
        'NegativeQuantityCount':     1,
        'ReturnRatio':               0.3,
        'TotalTransactions':         6,
        'Age':                       28.0,
        'SupportTicketsCount':       5,
        'SatisfactionScore':         2.0,
        'DaysSinceRegistration':     700,
        'RegMonth':                  1,
        'PreferredDayOfWeek':        1,
        'PreferredHour':             14,
        'UniqueCountries':           1,
        'AgeCategory':               '25-34',
        'SpendingCategory':          'Low',
        'LoyaltyLevel':              'Nouveau',
        'BasketSizeCategory':        'Petit',
        'IsPrivateIP':               0,
        'PreferredTimeOfDay':        'Après-midi',
        'Region':                    'UK',
        'WeekendPreference':         'Semaine',
        'ProductDiversity':          'Spécialisé',
        'Gender':                    'M',
        'AccountStatus':             'Active',
        'Country':                   'United Kingdom',
    }

    # Client fidèle (actif, dépenses élevées, satisfait)
    client_fidele = {
        'Frequency':                 35,
        'MonetaryTotal':             8500.0,
        'MonetaryAvg':               243.0,
        'AvgQuantityPerTransaction': 12.0,
        'WeekendPurchaseRatio':      0.15,
        'AvgDaysBetweenPurchases':   8.0,
        'UniqueProducts':            120,
        'AvgProductsPerTransaction': 6.5,
        'ZeroPriceCount':            0,
        'NegativeQuantityCount':     1,
        'ReturnRatio':               0.02,
        'TotalTransactions':         280,
        'Age':                       45.0,
        'SupportTicketsCount':       1,
        'SatisfactionScore':         5.0,
        'DaysSinceRegistration':     365,
        'RegMonth':                  6,
        'PreferredDayOfWeek':        2,
        'PreferredHour':             10,
        'UniqueCountries':           1,
        'AgeCategory':               '45-54',
        'SpendingCategory':          'VIP',
        'LoyaltyLevel':              'Ancien',
        'BasketSizeCategory':        'Grand',
        'IsPrivateIP':               0,
        'PreferredTimeOfDay':        'Matin',
        'Region':                    'UK',
        'WeekendPreference':         'Semaine',
        'ProductDiversity':          'Explorateur',
        'Gender':                    'F',
        'AccountStatus':             'Active',
        'Country':                   'United Kingdom',
    }

    for nom, client in [('Client à risque', client_a_risque),
                         ('Client fidèle',   client_fidele)]:
        pred = predict_client(client, art)
        print(f"\n{nom} :")
        print(f"  Churn prédit   : {'⚠️  Oui (parti)' if pred['churn'] else '✅ Non (fidèle)'}")
        print(f"  Probabilité    : {pred['probabilite']*100:.1f}%")
        print(f"  Niveau risque  : {pred['risque']}")
        print(f"  Recommandation : {pred['recommandation']}")

    print("\n→ Prochaine étape : app/ (interface Flask)")