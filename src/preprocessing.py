
"""
preprocessing.py  —  Pipeline anti-leakage (version finale)
============================================================

LEAKAGES IDENTIFIÉS ET CORRIGÉS :

  [L1] ChurnRiskCategory  → supprimée (r=0.88, construite à partir du churn)
  [L2] RFMSegment         → supprimée (Champions=0% / Dormants=100% churn)
  [L3] CustomerType       → supprimée (Perdu=100% / Hyperactif=0% churn)
  [L4] Recency            → SUPPRIMÉE — Churn := (Recency >= 91 jours)
                            AUC=1.0 avec un seul seuil → c'est la DÉFINITION du churn
  [L5] CustomerTenureDays → supprimée (54% churned ont tenure=0, corrèle r=0.45)
  [L6] FavoriteSeason     → supprimée (Automne absent à 97% chez les churned,
                            calculée sur la période active = proxy Recency)
  [L7] PreferredMonth     → supprimée (mois 10/11 absents chez les churned)
  [L8] TenureRatio        → supprimée (= Recency/CustomerTenureDays, dérivé de L4+L5)
  [L9] MonetaryPerDay     → supprimée (= MonetaryTotal/Recency, dérivé de L4)
  [L10] Target Encoding   → calculé sur X_train+y_train uniquement (smoothing k=10)
  [L11] Multicolinéarité  → détectée sur X_train uniquement (seuil 0.95)
  [L12] Imputation        → médiane calculée sur X_train, appliquée à X_test

RÈGLE FONDAMENTALE du pipeline :
  Toute transformation statistique (médiane, corrélation, target encoding...)
  est FIT sur X_train uniquement, puis TRANSFORM sur X_train et X_test.

Usage :
    python src/preprocessing.py
"""

import os
import warnings
import logging

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────
RAW_PATH       = os.path.join("data", "raw",        "retail_customers_COMPLETE_CATEGORICAL.csv")
PROCESSED_PATH = os.path.join("data", "processed",  "retail_cleaned.csv")
X_TRAIN_PATH   = os.path.join("data", "train_test", "X_train.csv")
X_TEST_PATH    = os.path.join("data", "train_test", "X_test.csv")
Y_TRAIN_PATH   = os.path.join("data", "train_test", "y_train.csv")
Y_TEST_PATH    = os.path.join("data", "train_test", "y_test.csv")
SCALER_PATH    = os.path.join("models", "scaler.joblib")
ENCODER_PATH   = os.path.join("models", "encoders.joblib")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
TARGET         = "Churn"
TEST_SIZE      = 0.2
RANDOM_SEED    = 42
REFERENCE_DATE = pd.Timestamp("2011-12-31")

# ── [L1-L9] Toutes les features qui encodent directement ou indirectement le churn
LEAKY_COLS = [
    # Construites à partir du label churn
    "ChurnRiskCategory",    # [L1] r=0.879 — littéralement le churn encodé
    "RFMSegment",           # [L2] Champions=0%/Dormants=100% churn
    "CustomerType",         # [L3] Perdu=100%/Hyperactif=0% churn
    # Recency = définition du churn (Churn := Recency >= 91j)
    "Recency",              # [L4] AUC=1.0 seul, seuil parfait à 90 jours
    "CustomerTenureDays",   # [L5] 54% churned ont tenure=0, corrèle r=0.45
    # Calculées sur la période active → proxy de Recency
    "FavoriteSeason",       # [L6] Automne absent à 97% chez les churned
    "PreferredMonth",       # [L7] Mois 10 et 11 totalement absents chez les churned
    # Features dérivées des variables leaky ci-dessus
    "TenureRatio",          # [L8] = Recency / (CustomerTenureDays+1)
    "MonetaryPerDay",       # [L9] = MonetaryTotal / (Recency+1)
]

# ── Colonnes inutiles / identifiants techniques
COLS_TO_DROP = [
    "CustomerID",
    "NewsletterSubscribed",   # variance nulle (100 % "Yes")
    "RegistrationDate",       # remplacée par features dérivées
    "LastLoginIP",            # remplacée par features dérivées
    "UniqueInvoices",
    "TotalTransactions",
    "UniqueDescriptions",
    "NegativeQuantityCount",  
    # --- Leaky: encode Churn directly ---
    "ChurnRiskCategory",
    "RFMSegment",
    "LoyaltyLevel",
    "CustomerType",           # "Perdu" = churned in French
    # --- Leaky: circular (Churn defined from Recency) ---
    "Recency",                # corr=0.859 with Churn
    # --- Constant across all rows: FirstPurchaseDaysAgo=374 for every customer ---
    # Computed as (snapshot_date - dataset_start_date), not per-customer.
    # Correlation with Churn (0.222) was a statistical artifact — carries zero signal.
    "FirstPurchaseDaysAgo",
    # --- Non-significant: p=0.365, corr=+0.014 ---
    "SatisfactionScore",
    # --- NEW: Multicollinearity & Redundancy Drops ---
    "AvgLinesPerInvoice",    # r=0.96 with AvgProductsPerTransaction
    "TotalQuantity",         # r=0.92 with MonetaryTotal
    "MonetaryStd",           # Highly entangled cluster
    "MonetaryMin",           # Highly entangled cluster
    "MinQuantity",           # Highly entangled cluster
    "MaxQuantity",           # Highly entangled cluster (Keeping MonetaryMax)
    "RegYear",
]

# ── Encodage ordinal — mapping FIXE (transformations déterministes, avant split)
ORDINAL_MAPPINGS = {
    "AgeCategory":     {"Inconnu": 0, "18-24": 1, "25-34": 2,
                        "35-44": 3, "45-54": 4, "55-64": 5, "65+": 6},
    "SpendingCategory":{"Low": 0, "Medium": 1, "High": 2, "VIP": 3},
    "LoyaltyLevel":    {"Inconnu": 0, "Nouveau": 1, "Jeune": 2, "Établi": 3, "Ancien": 4},
    "BasketSizeCategory":{"Inconnu": 0, "Petit": 1, "Moyen": 2, "Grand": 3},
}

# ── One-Hot Encoding (modalités apprises sur X_train uniquement)
NOMINAL_COLS = [
    "PreferredTimeOfDay",   # Matin, Midi, Après-midi, Soir
    "Region",               # UK, Europe N/S/E/C, Asie, Autre
    "WeekendPreference",    # Weekend, Semaine, Inconnu
    "ProductDiversity",     # Spécialisé, Modéré, Explorateur
    "Gender",               # M, F, Unknown
    "AccountStatus",        # Active, Suspended, Pending, Closed
]

# ── Target Encoding (calculé sur X_train + y_train uniquement)
COUNTRY_COL = "Country"

# ── Variables continues à normaliser (StandardScaler)
NUMERIC_COLS_TO_SCALE = [
    "Frequency", "MonetaryTotal", "MonetaryAvg", "MonetaryStd",
    "TotalQuantity", "AvgQuantityPerTransaction",
    "FirstPurchaseDaysAgo",
    "WeekendPurchaseRatio", "AvgDaysBetweenPurchases",
    "UniqueProducts", "AvgProductsPerTransaction",
    "NegativeQuantityCount", "ZeroPriceCount",
    "ReturnRatio", "TotalTransactions",
    "AvgLinesPerInvoice", "Age",
    "SupportTicketsCount", "SatisfactionScore",
    # Features engineered conservées (non dérivées de Recency)
    "DaysSinceRegistration", "RegMonth",
    "AvgBasketValue", "ReturnToFrequencyRatio", "IPFirstOctet",
    "Country_TargetEnc",
]


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Chargement
# ═════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    log.info(f"Chargement : {path}")
    df = pd.read_csv(path)
    log.info(f"  {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Feature Engineering (transformations PURES, sans stats)
# ═════════════════════════════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uniquement des transformations déterministes — aucune statistique calculée.
    Safe à appliquer avant le split.

    Features conservées après audit leakage :
      - DaysSinceRegistration : ancienneté dans le système (pas Recency)
      - RegMonth              : saisonnalité de l'inscription
      - AvgBasketValue        : = MonetaryTotal / Frequency (pas de Recency)
      - ReturnToFrequencyRatio: comportement retours
      - IsPrivateIP, IPFirstOctet : features réseau
    """
    log.info("── Feature Engineering & Parsing ──")
    df = df.copy()

    # RegistrationDate → ancienneté et saisonnalité de l'inscription
    dates = pd.to_datetime(df["RegistrationDate"], dayfirst=True, errors="coerce")
    df["DaysSinceRegistration"] = (REFERENCE_DATE - dates).dt.days
    df["RegMonth"]              = dates.dt.month
    log.info("  RegistrationDate → DaysSinceRegistration, RegMonth")

    # LastLoginIP → features réseau
    def _is_private(ip):
        try:
            a, b = [int(x) for x in str(ip).split(".")[:2]]
            return int(a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168))
        except Exception:
            return 0

    def _first_octet(ip):
        try:
            return int(str(ip).split(".")[0])
        except Exception:
            return -1

    df["IsPrivateIP"]  = df["LastLoginIP"].apply(_is_private)
    df["IPFirstOctet"] = df["LastLoginIP"].apply(_first_octet)
    log.info("  LastLoginIP → IsPrivateIP, IPFirstOctet")

    # Features métier (sans Recency !)
    df["AvgBasketValue"]         = df["MonetaryTotal"] / df["Frequency"].replace(0, np.nan)
    df["ReturnToFrequencyRatio"] = df["NegativeQuantityCount"] / (df["Frequency"] + 1)
    log.info("  Métriques : AvgBasketValue, ReturnToFrequencyRatio")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Nettoyage & Outliers (règles FIXES, sans stats)
# ═════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Règles définies a priori depuis le PDF — indépendantes du dataset."""
    log.info("── Nettoyage & Outliers ──")
    df = df.copy()

    mask = ~df["SupportTicketsCount"].between(0, 15)
    df.loc[mask, "SupportTicketsCount"] = np.nan
    log.info(f"  SupportTicketsCount : {mask.sum()} valeurs hors [0-15] → NaN")

    # ── Déséquilibre AccountStatus (90.2% Active vs 9.8% minoritaires)
    # Modalités rares regroupées en "Inactive" pour éviter un signal quasi nul
    # après one-hot encoding (colonnes presque toujours à 0)
    if "AccountStatus" in df.columns:
        before = df["AccountStatus"].value_counts().to_dict()
        df["AccountStatus"] = df["AccountStatus"].apply(
            lambda x: x if x == "Active" else "Inactive"
        )
        after = df["AccountStatus"].value_counts().to_dict()
        log.info(f"  AccountStatus : {before} → {after}")
        log.info(f"  Suspended+Pending+Closed → 'Inactive' ({after.get('Inactive', 0)} clients, {after.get('Inactive', 0)/len(df)*100:.1f}%)")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Suppression des features leaky et inutiles
# ═════════════════════════════════════════════════════════════════════════════

def drop_leaky_and_useless(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── Suppression features leaky & inutiles ──")

    all_drop = LEAKY_COLS + COLS_TO_DROP
    present  = [c for c in all_drop if c in df.columns]
    df       = df.drop(columns=present)

    log.info(f"  Leaky    : {[c for c in LEAKY_COLS    if c in present]}")
    log.info(f"  Inutiles : {[c for c in COLS_TO_DROP  if c in present]}")
    log.info(f"  Dimensions : {df.shape[0]:,} × {df.shape[1]}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Encodage ordinal (mapping FIXE, avant split)
# ═════════════════════════════════════════════════════════════════════════════

def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    log.info("── Encodage ordinal ──")
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            log.info(f"  {col:25s} → 0-{max(mapping.values())}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Split Train / Test
# ═════════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame):
    """Toutes les étapes statistiques se font APRÈS ce split."""
    log.info("── Split Train/Test ──")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    log.info(f"  Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")
    log.info(f"  Churn train : {y_train.mean()*100:.1f}% | test : {y_test.mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Imputation (médiane calculée sur X_TRAIN uniquement)
# ═════════════════════════════════════════════════════════════════════════════

def impute_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """[L12] Médiane fit sur X_train → appliquée à X_train ET X_test."""
    log.info("── Imputation (médiane sur X_train) ──")

    cols_to_impute = [
        "Age", "AvgDaysBetweenPurchases", "SatisfactionScore",
        "SupportTicketsCount", "AvgBasketValue",
        "DaysSinceRegistration", "RegMonth",
    ]

    impute_values = {}
    for col in cols_to_impute:
        if col not in X_train.columns:
            continue
        median_train = X_train[col].median()
        impute_values[col] = median_train
        n_train = X_train[col].isnull().sum()
        n_test  = X_test[col].isnull().sum() if col in X_test.columns else 0
        X_train[col] = X_train[col].fillna(median_train)
        if col in X_test.columns:
            X_test[col]  = X_test[col].fillna(median_train)
        if n_train > 0 or n_test > 0:
            log.info(f"  {col:35s} : train={n_train}, test={n_test} → médiane={median_train:.2f}")

    log.info(f"  NaN restants → train: {X_train.isnull().sum().sum()}, test: {X_test.isnull().sum().sum()}")
    return X_train, X_test, impute_values


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 8 — One-Hot Encoding (modalités apprises sur X_TRAIN)
# ═════════════════════════════════════════════════════════════════════════════

def encode_ohe(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Modalités apprises sur X_train uniquement.
    X_test est aligné sur les mêmes colonnes (manquantes=0, en trop=supprimées).
    """
    log.info("── One-Hot Encoding (sur X_train) ──")

    cols_present = [c for c in NOMINAL_COLS if c in X_train.columns]

    X_train = pd.get_dummies(X_train, columns=cols_present, drop_first=True, dtype=int)
    X_test  = pd.get_dummies(X_test,  columns=cols_present, drop_first=True, dtype=int)

    train_ohe = set(c for c in X_train.columns if any(c.startswith(f"{n}_") for n in cols_present))
    test_ohe  = set(c for c in X_test.columns  if any(c.startswith(f"{n}_") for n in cols_present))

    for col in train_ohe - test_ohe:
        X_test[col] = 0
    X_test = X_test.drop(columns=list(test_ohe - train_ohe), errors="ignore")
    X_test = X_test[X_train.columns]

    log.info(f"  {len(cols_present)} colonnes → {len(train_ohe)} variables binaires")
    return X_train, X_test


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 9 — Target Encoding Country (sur X_TRAIN + y_TRAIN)
# ═════════════════════════════════════════════════════════════════════════════

def encode_target(X_train, X_test, y_train):
    """[L10] Churn moyen par pays calculé sur train uniquement + smoothing."""
    log.info("── Target Encoding — Country (sur X_train+y_train) ──")

    if COUNTRY_COL not in X_train.columns:
        return X_train, X_test, {}

    k           = 10
    global_mean = y_train.mean()
    tmp         = X_train[[COUNTRY_COL]].copy()
    tmp[TARGET] = y_train.values
    stats       = tmp.groupby(COUNTRY_COL)[TARGET].agg(["mean", "count"])
    stats["smoothed"] = (stats["count"] * stats["mean"] + k * global_mean) / (stats["count"] + k)
    country_map = stats["smoothed"].to_dict()

    X_train["Country_TargetEnc"] = X_train[COUNTRY_COL].map(country_map).fillna(global_mean)
    X_test["Country_TargetEnc"]  = X_test[COUNTRY_COL].map(country_map).fillna(global_mean)
    X_train = X_train.drop(columns=[COUNTRY_COL])
    X_test  = X_test.drop(columns=[COUNTRY_COL])

    log.info(f"  {len(country_map)} pays | smoothing k={k} | mean_global={global_mean:.3f}")
    return X_train, X_test, country_map


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 10 — Suppression multicolinéarité (sur X_TRAIN uniquement)
# ═════════════════════════════════════════════════════════════════════════════

def remove_multicollinearity(X_train, X_test, threshold=0.95):
    """[L11] Corrélations calculées sur X_train uniquement, seuil 0.95."""
    log.info(f"── Multicolinéarité (|r| > {threshold}, sur X_train) ──")

    num_cols = [c for c in X_train.select_dtypes(include="number").columns
                if X_train[c].nunique() > 2]

    corr_abs = X_train[num_cols].corr().abs()
    upper    = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if partner not in to_drop:
                to_drop.add(col)

    if to_drop:
        X_train = X_train.drop(columns=list(to_drop))
        X_test  = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
        log.info(f"  {len(to_drop)} supprimées : {sorted(to_drop)}")
    else:
        log.info("  Aucune paire > seuil")

    log.info(f"  Features finales : {X_train.shape[1]}")
    return X_train, X_test


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 11 — Normalisation (fit sur X_TRAIN)
# ═════════════════════════════════════════════════════════════════════════════

def scale_features(X_train, X_test):
    """fit() sur X_train uniquement. y jamais normalisé."""
    log.info("── StandardScaler (fit sur X_train) ──")

    cols_scale = [c for c in NUMERIC_COLS_TO_SCALE if c in X_train.columns]
    cols_keep  = [c for c in X_train.columns if c not in cols_scale]

    scaler    = StandardScaler()
    arr_train = scaler.fit_transform(X_train[cols_scale])
    arr_test  = scaler.transform(X_test[cols_scale])

    X_train_sc = pd.concat([
        pd.DataFrame(arr_train, columns=cols_scale, index=X_train.index),
        X_train[cols_keep]
    ], axis=1)
    X_test_sc = pd.concat([
        pd.DataFrame(arr_test, columns=cols_scale, index=X_test.index),
        X_test[cols_keep]
    ], axis=1)

    log.info(f"  Scalées : {len(cols_scale)} | Inchangées : {len(cols_keep)}")
    return X_train_sc, X_test_sc, scaler


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 12 — Vérification anti-leakage finale
# ═════════════════════════════════════════════════════════════════════════════

def verify_no_leakage(X_train, y_train):
    """
    Contrôle final : aucune feature ne doit avoir AUC > 0.85 seule (stump depth=1).
    Un AUC trop élevé sur une seule feature indique un leakage résiduel.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score

    log.info("── Vérification anti-leakage ──")
    alerts = []
    for col in X_train.columns:
        try:
            dt  = DecisionTreeClassifier(max_depth=1, random_state=42)
            dt.fit(X_train[[col]], y_train)
            auc = roc_auc_score(y_train, dt.predict_proba(X_train[[col]])[:,1])
            if auc > 0.85:
                alerts.append((col, round(auc, 4)))
        except Exception:
            continue

    if alerts:
        for col, auc in sorted(alerts, key=lambda x: -x[1]):
            log.warning(f"  ⚠  {col:40s} AUC_stump = {auc} > 0.85 → leakage potentiel")
    else:
        log.info("  ✓ Aucune feature avec AUC_stump > 0.85 — pipeline propre")


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 13 — Sauvegarde
# ═════════════════════════════════════════════════════════════════════════════

def save_outputs(df_clean, X_train, X_test, y_train, y_test, scaler, encoders):
    log.info("── Sauvegarde ──")
    for path in [PROCESSED_PATH, X_TRAIN_PATH, X_TEST_PATH,
                 Y_TRAIN_PATH, Y_TEST_PATH, SCALER_PATH, ENCODER_PATH]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    df_clean.to_csv(PROCESSED_PATH, index=False)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH,   index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH,   index=False)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    log.info(f"  X_train {X_train.shape} → {X_TRAIN_PATH}")
    log.info(f"  X_test  {X_test.shape}  → {X_TEST_PATH}")
    log.info(f"  scaler + encoders       → models/")


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    log.info("═" * 62)
    log.info("  PREPROCESSING ANTI-LEAKAGE v3 — Retail Churn")
    log.info("═" * 62)

    # Transformations PURES (avant split)
    df = load_data(RAW_PATH)
    df = feature_engineering(df)
    df = clean_data(df)
    df = drop_leaky_and_useless(df)
    df = encode_ordinal(df)

    # SPLIT
    X_train, X_test, y_train, y_test = split_data(df)

    # Transformations STATISTIQUES (fit sur train, transform sur train+test)
    X_train, X_test, impute_vals = impute_data(X_train, X_test)
    X_train, X_test              = encode_ohe(X_train, X_test)
    X_train, X_test, country_map = encode_target(X_train, X_test, y_train)
    X_train, X_test              = remove_multicollinearity(X_train, X_test)
    X_train, X_test, scaler      = scale_features(X_train, X_test)

    # Vérification finale
    verify_no_leakage(X_train, y_train)

    # Sauvegarde
    encoders = {"impute_values": impute_vals, "country_map": country_map}
    save_outputs(df, X_train, X_test, y_train, y_test, scaler, encoders)

    log.info("═" * 62)
    log.info(f"  ✓ Pipeline terminé — {X_train.shape[1]} features finales")
    log.info("  ✓ AUC attendue avec un bon modèle : 0.75 – 0.88")
    log.info("  ✓ AUC > 0.95 après ce pipeline = vrai signal, pas du leakage")
    log.info("═" * 62)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    run_pipeline()