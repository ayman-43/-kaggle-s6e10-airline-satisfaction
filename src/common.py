"""Shared data loading, feature engineering and CV setup for playground-series-s6e10."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OOF_DIR = ROOT / "oof"
SUB_DIR = ROOT / "submissions"
OOF_DIR.mkdir(exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)

TARGET = "satisfaction"
N_FOLDS = 5
SEED = 42

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
BASE_FEATURES = NUM_COLS + RATING_COLS + CAT_COLS


def load():
    """Return (train, test, original) with a binary integer target and identical columns."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    orig = pd.concat([pd.read_csv(DATA / "original" / f) for f in ("train.csv", "test.csv")],
                     ignore_index=True)
    orig = orig.drop(columns=["Unnamed: 0", "id", "Inflight service"])
    orig[TARGET] = (orig[TARGET] == "satisfied").astype(int)
    train[TARGET] = train[TARGET].astype(int)
    return train, test, orig


def add_features(df):
    df = df.copy()
    r = df[RATING_COLS]
    df["rating_sum"] = r.sum(axis=1)
    df["rating_std"] = r.std(axis=1)
    df["n_zero"] = (r == 0).sum(axis=1)
    df["n_five"] = (r == 5).sum(axis=1)
    df["n_low"] = (r <= 2).sum(axis=1)
    df["delay_total"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["delay_diff"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]
    df["arr_na"] = df["Arrival Delay in Minutes"].isna().astype(int)
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df


ENG_FEATURES = ["rating_sum", "rating_std", "n_zero", "n_five", "n_low",
                "delay_total", "delay_diff", "arr_na"]


FD = "Flight Distance"
# Flight Distance acts as a route id; these keys get in-fold target encoding (see train.py --v2).
TE_KEYS = [[FD], ["Age"], [FD, "Age"], [FD, "Class"], [FD, "Type of Travel"],
           ["Age", "Class", "Type of Travel", "Customer Type"]]
SEGMENT = ["Type of Travel", "Class", "Customer Type"]
# v3: route groups, route x each rating, passenger segment x each rating
TE_KEYS_V3 = (TE_KEYS + [["fd10"], ["fd10", "Class"], SEGMENT]
              + [[FD, r] for r in RATING_COLS] + [SEGMENT + [r] for r in RATING_COLS])


def v2_features(train, test, orig, v3=False):
    """Target-free extra features for train/test, plus integer key codes for target encoding.

    - og_lgb: logit of a LightGBM trained only on the original dataset
    - cnt_*: how often each numeric value appears in train+test
    - fdm_*, fd_cnt: per-Flight-Distance (route) mean of every other column
    v3 adds og_xgb (XGBoost on the original data) and the wider TE_KEYS_V3.
    """
    import lightgbm as lgb
    full = pd.concat([train[BASE_FEATURES], test[BASE_FEATURES]], ignore_index=True)
    extra = pd.DataFrame(index=full.index)

    m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=63, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=-1,
                           random_state=0)
    m.fit(orig[BASE_FEATURES], orig[TARGET])
    p = np.clip(m.predict_proba(full)[:, 1], 1e-6, 1 - 1e-6)
    extra["og_lgb"] = np.log(p / (1 - p)).astype("float32")
    if v3:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
                              colsample_bytree=0.5, enable_categorical=True, tree_method="hist",
                              device="cuda", random_state=0)
        m.fit(orig[BASE_FEATURES], orig[TARGET])
        p = np.clip(m.predict_proba(full)[:, 1], 1e-6, 1 - 1e-6)
        extra["og_xgb"] = np.log(p / (1 - p)).astype("float32")

    for c in NUM_COLS:
        extra["cnt_" + c] = full[c].map(full[c].value_counts()).fillna(0).astype("float32")

    coded = full.copy()
    for c in CAT_COLS:
        coded[c] = coded[c].cat.codes
    grp = coded.groupby(FD)
    for c in BASE_FEATURES:
        if c != FD:
            extra["fdm_" + c] = grp[c].transform("mean").astype("float32")
    extra["fd_cnt"] = grp["Age"].transform("size").astype("float32")

    keys = pd.DataFrame(index=full.index)
    full = full.assign(fd10=full[FD] // 10)
    for cols in (TE_KEYS_V3 if v3 else TE_KEYS):
        s = full[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "|" + full[c].astype(str)
        keys["te_" + "|".join(cols)] = pd.factorize(s)[0]

    n = len(train)
    return (extra.iloc[:n].reset_index(drop=True), extra.iloc[n:].reset_index(drop=True),
            keys.iloc[:n].reset_index(drop=True), keys.iloc[n:].reset_index(drop=True))


def aux_features(kind, n_train):
    """Auxiliary-task features from data/aux.parquet (see aux_features.py), split into (train, test).

    kind: "ev" = the 13 expected ratings, "all" = every aux_* column, "allog" = aux_* + ogaux_*,
    "ogev" = the 13 original-data expected ratings.
    """
    a = pd.read_parquet(DATA / "aux.parquet")
    pick = {"ev": lambda c: c.startswith("aux_ev_"), "all": lambda c: c.startswith("aux_"),
            "allog": lambda c: True, "ogev": lambda c: c.startswith("ogaux_ev_")}[kind]
    a = a[[c for c in a.columns if pick(c)]]
    return a.iloc[:n_train].reset_index(drop=True), a.iloc[n_train:].reset_index(drop=True)


def folds(y):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def save(name, oof, test_pred, test_ids):
    np.save(OOF_DIR / f"{name}_oof.npy", oof)
    np.save(OOF_DIR / f"{name}_test.npy", test_pred)
    pd.DataFrame({"id": test_ids, TARGET: test_pred}).to_csv(SUB_DIR / f"{name}.csv", index=False)
