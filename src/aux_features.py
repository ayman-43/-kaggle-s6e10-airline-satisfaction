"""Auxiliary-task features: predict each rating / category column from the other 20 columns.

No competition label is used, so the features are computed once for train + test and are leak-free.
  aux_p_<c>      cross-fitted probability of the row's actual value of <c> (train+test, KFold 5)
  aux_ev_<c>     expected value of rating <c> given the other columns
  aux_sum_logp   row log-likelihood (sum of log aux_p_*)
  ogaux_*        the same, from predictors trained only on the original dataset
Idea from goodpjw2008's "S6E10 auxiliary-task features" notebook.

Usage: python src/aux_features.py   -> data/aux.parquet
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from common import BASE_FEATURES, CAT_COLS, DATA, RATING_COLS, load

AUX_TARGETS = RATING_COLS + ["Class", "Type of Travel", "Customer Type"]

train, test, orig = load()
full = pd.concat([train[BASE_FEATURES], test[BASE_FEATURES]], ignore_index=True)
both = pd.concat([full, orig[BASE_FEATURES]], ignore_index=True)
for c in CAT_COLS:
    both[c] = both[c].astype("category").cat.codes
coded, og = both.iloc[:len(full)].reset_index(drop=True), both.iloc[len(full):].reset_index(drop=True)


def xgb_for(k, n_est, lr, seed):
    return xgb.XGBClassifier(n_estimators=n_est, learning_rate=lr, max_depth=6, subsample=0.8,
                             colsample_bytree=0.7, tree_method="hist", device="cuda", random_state=seed,
                             objective="multi:softprob" if k > 2 else "binary:logistic")


out = pd.DataFrame(index=range(len(full)))
t0 = time.time()
for c in AUX_TARGETS:
    others = [f for f in BASE_FEATURES if f != c]
    vals = np.sort(coded[c].unique())
    yc = np.searchsorted(vals, coded[c].values)
    P = np.zeros((len(full), len(vals)), dtype=np.float32)
    for a, b in KFold(5, shuffle=True, random_state=0).split(coded):
        m = xgb_for(len(vals), 300, 0.1, 0).fit(coded.iloc[a][others], yc[a])
        P[b] = m.predict_proba(coded.iloc[b][others])
    out[f"aux_p_{c}"] = P[np.arange(len(full)), yc]
    if c in RATING_COLS:
        out[f"aux_ev_{c}"] = P @ vals.astype(np.float32)

    # original-data predictors: classes are the values seen in the original data
    ovals = np.sort(og[c].unique())
    Q = np.zeros((len(full), len(ovals)))
    for s in range(3):
        m = xgb_for(len(ovals), 400, 0.05, s).fit(og[others], np.searchsorted(ovals, og[c].values))
        Q += m.predict_proba(coded[others]) / 3
    idx = np.searchsorted(ovals, coded[c].values).clip(0, len(ovals) - 1)
    out[f"ogaux_p_{c}"] = Q[np.arange(len(full)), idx]
    if c in RATING_COLS:
        out[f"ogaux_ev_{c}"] = Q @ ovals
    print(f"{c:36s} acc {(P.argmax(1) == yc).mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)

for pre in ("aux", "ogaux"):
    cols = [c for c in out if c.startswith(pre + "_p_")]
    out[pre + "_sum_logp"] = np.log(np.clip(out[cols].values, 1e-6, 1)).sum(1)
out = out.astype(np.float32)
out.to_parquet(DATA / "aux.parquet")
print(out.shape, "->", DATA / "aux.parquet")
