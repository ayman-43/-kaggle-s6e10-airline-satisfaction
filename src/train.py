"""Train one model with 5-fold CV, optionally appending the original dataset to each training fold.

Usage: python src/train.py <lgb|xgb|cat> [--no-orig] [--no-eng] [--as-cat] [--v2] [--name NAME]
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import TargetEncoder

from common import (BASE_FEATURES, CAT_COLS, ENG_FEATURES, N_FOLDS, TARGET, add_features,
                    aux_features, folds, load, save, v2_features)

p = argparse.ArgumentParser()
p.add_argument("model", choices=["lgb", "xgb", "cat"])
p.add_argument("--no-orig", action="store_true")
p.add_argument("--no-eng", action="store_true")
p.add_argument("--as-cat", action="store_true",
               help="lgb/xgb: add categorical copies of every numeric raw column")
p.add_argument("--v2", action="store_true",
               help="add original-model logit, route profiles, counts and in-fold target encoding "
                    "(implies --no-orig)")
p.add_argument("--v3", action="store_true", help="v2 + og_xgb + wider target-encoding keys")
p.add_argument("--aux", choices=["ev", "all", "allog", "ogev"],
               help="append auxiliary-task features (data/aux.parquet)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-bin", type=int, default=256)
p.add_argument("--no-te", action="store_true", help="v2/v3 features without our target encoding")
p.add_argument("--l2", type=float, default=3.0, help="catboost l2_leaf_reg")
p.add_argument("--depth", type=int, default=7, help="catboost depth")
p.add_argument("--name")
args = p.parse_args()
if args.v3:
    args.v2 = True
if args.v2:
    args.no_orig = True

train, test, orig = load()
train, test, orig = add_features(train), add_features(test), add_features(orig)
features = BASE_FEATURES + ([] if args.no_eng else ENG_FEATURES)
if args.as_cat:
    for c in BASE_FEATURES:
        if c in CAT_COLS:
            continue
        as_int = [df[c].fillna(-1).astype(int) for df in (train, test, orig)]
        cats = np.unique(pd.concat(as_int))
        for df, v in zip((train, test, orig), as_int):
            df[c + "_c"] = pd.Categorical(v, categories=cats)
        features.append(c + "_c")
y = train[TARGET].values
te_cols = []
if args.v2:
    ex_tr, ex_te, keys_tr, keys_te = v2_features(train, test, orig, v3=args.v3)
    train = pd.concat([train, ex_tr], axis=1)
    test = pd.concat([test, ex_te], axis=1)
    features += list(ex_tr.columns)
    te_cols = [] if args.no_te else list(keys_tr.columns)
if args.aux:
    ax_tr, ax_te = aux_features(args.aux, len(train))
    train, test = pd.concat([train, ax_tr], axis=1), pd.concat([test, ax_te], axis=1)
    features += list(ax_tr.columns)


def add_te(tr_idx, va_idx):
    """In-fold target encoding: training rows are cross-fitted, so no row sees its own label."""
    X_tr, X_va, X_te = (train.iloc[tr_idx][features].copy(), train.iloc[va_idx][features].copy(),
                        test[features].copy())
    if te_cols:
        enc = TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True,
                            random_state=0)
        X_tr[te_cols] = enc.fit_transform(keys_tr.iloc[tr_idx], y[tr_idx])
        X_va[te_cols] = enc.transform(keys_tr.iloc[va_idx])
        X_te[te_cols] = enc.transform(keys_te)
    return X_tr, X_va, X_te


def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
    if args.model == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            n_estimators=10000, learning_rate=0.02, num_leaves=63, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3.0,
            max_cat_to_onehot=4, cat_smooth=20, cat_l2=10, verbose=-1, random_state=args.seed,
            n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1], m.best_iteration_
    if args.model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(
            n_estimators=10000, learning_rate=0.02, max_depth=8, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=3.0, tree_method="hist",
            max_bin=args.max_bin,
            device="cuda", enable_categorical=True, max_cat_to_onehot=4, eval_metric="auc",
            early_stopping_rounds=300, random_state=args.seed)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1], m.best_iteration
    from catboost import CatBoostClassifier, Pool
    # CatBoost: treat every raw column as categorical too, which captures synthetic-data artifacts.
    def prep(X):
        X = X.copy()
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        for c in BASE_FEATURES:
            if c not in CAT_COLS:
                X[c + "_c"] = X[c].fillna(-1).astype(int).astype(str)
        return X
    X_tr, X_va, X_te = prep(X_tr), prep(X_va), prep(X_te)
    cats = CAT_COLS + [c for c in X_tr.columns if c.endswith("_c")]
    m = CatBoostClassifier(
        iterations=10000, learning_rate=0.08, depth=args.depth, l2_leaf_reg=args.l2, eval_metric="AUC",
        task_type="GPU", metric_period=50, early_stopping_rounds=300, random_seed=args.seed, verbose=0)
    m.fit(Pool(X_tr, y_tr, cat_features=cats), eval_set=Pool(X_va, y_va, cat_features=cats))
    return (m.predict_proba(Pool(X_va, cat_features=cats))[:, 1],
            m.predict_proba(Pool(X_te, cat_features=cats))[:, 1], m.best_iteration_)


oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
t0 = time.time()
for k, (tr_idx, va_idx) in enumerate(folds(y)):
    X_tr, X_va, X_te = add_te(tr_idx, va_idx)
    y_tr = y[tr_idx]
    if not args.no_orig:  # original data only ever goes into training, never validation
        X_tr = pd.concat([X_tr, orig[features]], ignore_index=True)
        y_tr = np.concatenate([y_tr, orig[TARGET].values])
    va_pred, te_pred, it = fit_predict(X_tr, y_tr, X_va, y[va_idx], X_te)
    oof[va_idx] = va_pred
    test_pred += te_pred / N_FOLDS
    print(f"fold {k}: auc={roc_auc_score(y[va_idx], va_pred):.5f} iters={it} "
          f"({time.time() - t0:.0f}s)", flush=True)

name = args.name or args.model
print(f"{name} CV AUC: {roc_auc_score(y, oof):.5f}")
save(name, oof, test_pred, test["id"].values)
