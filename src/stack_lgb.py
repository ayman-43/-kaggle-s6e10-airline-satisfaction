"""Level-2 LightGBM stacker on OOF logits (+ base features), blended with the LR stack.

Usage: python src/stack_lgb.py name1 name2 ... [--public MIN_AUC] [--C 0.05] [--out stack_l2]
Writes submissions/<out>_lgb.csv and <out>_blend.csv (rank blend at the CV-best weight).
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import BASE_FEATURES, CAT_COLS, OOF_DIR, SUB_DIR, TARGET, folds, load

p = argparse.ArgumentParser()
p.add_argument("names", nargs="+")
p.add_argument("--out", default="stack_l2")
p.add_argument("--C", type=float, default=0.05)
p.add_argument("--public", type=float, default=None)
args = p.parse_args()

train, test, _ = load()
y = train[TARGET].values


def score(path):
    a = np.load(path, allow_pickle=False).astype("float64").ravel()
    if a.min() < 0 or a.max() > 1:  # some public files are already logits
        return a
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


oof_paths = [OOF_DIR / f"{n}_oof.npy" for n in args.names]
test_paths = [OOF_DIR / f"{n}_test.npy" for n in args.names]
if args.public is not None:
    pub = pd.read_csv(OOF_DIR.parent / "data" / "public_oof" / "survey.csv")
    pub = pub[pub.auc >= args.public]
    oof_paths += list(pub.oof)
    test_paths += list(pub.test)
X = np.column_stack([score(f) for f in oof_paths]).astype("float32")
X_te = np.column_stack([score(f) for f in test_paths]).astype("float32")
mu, sd = X.mean(0), X.std(0)
X, X_te = (X - mu) / sd, (X_te - mu) / sd
print("models:", X.shape[1])

raw = pd.concat([train[BASE_FEATURES], test[BASE_FEATURES]])
for c in CAT_COLS:
    raw[c] = raw[c].astype("category").cat.codes
raw = raw.values.astype("float32")
Z, Z_te = np.hstack([X, raw[: len(y)]]), np.hstack([X_te, raw[len(y):]])

params = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              verbose=-1, n_jobs=-1)
oof_lr, oof_gb = np.zeros(len(y)), np.zeros(len(y))
iters = []
for k, (tr, va) in enumerate(folds(y)):
    oof_lr[va] = LogisticRegression(C=args.C, max_iter=1000).fit(X[tr], y[tr]).decision_function(X[va])
    m = lgb.train(params, lgb.Dataset(Z[tr], y[tr]), 3000, valid_sets=[lgb.Dataset(Z[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_gb[va] = m.predict(Z[va], raw_score=True)
    iters.append(m.best_iteration)
    print(f"fold {k}: lr={roc_auc_score(y[va], oof_lr[va]):.5f} lgb={roc_auc_score(y[va], oof_gb[va]):.5f} it={m.best_iteration}", flush=True)
print(f"LR stack CV {roc_auc_score(y, oof_lr):.5f}  LGB stack CV {roc_auc_score(y, oof_gb):.5f}")

r_lr, r_gb = rankdata(oof_lr) / len(y), rankdata(oof_gb) / len(y)
ws = np.round(np.arange(0, 1.01, 0.1), 1)
cv = [roc_auc_score(y, (1 - w) * r_lr + w * r_gb) for w in ws]
for w, s in zip(ws, cv):
    print(f"w_lgb={w:.1f} CV {s:.5f}")
w = ws[int(np.argmax(cv))]

pred_lr = LogisticRegression(C=args.C, max_iter=1000).fit(X, y).decision_function(X_te)
pred_gb = lgb.train(params, lgb.Dataset(Z, y), int(np.mean(iters) * 1.1)).predict(Z_te, raw_score=True)
t_lr, t_gb = rankdata(pred_lr) / len(pred_lr), rankdata(pred_gb) / len(pred_gb)
pd.DataFrame({"id": test["id"], TARGET: t_gb}).to_csv(SUB_DIR / f"{args.out}_lgb.csv", index=False)
pd.DataFrame({"id": test["id"], TARGET: (1 - w) * t_lr + w * t_gb}).to_csv(SUB_DIR / f"{args.out}_blend.csv", index=False)
print(f"wrote {args.out}_lgb.csv and {args.out}_blend.csv (w_lgb={w})")
