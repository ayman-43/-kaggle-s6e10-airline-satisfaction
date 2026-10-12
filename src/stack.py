"""Stack saved models with a logistic regression on their OOF logits (scored with the same 5 folds).

Usage: python src/stack.py name1 name2 ... [--out stack] [--C 1.0] [--public MIN_AUC]
--public also stacks the downloaded public OOFs (data/public_oof/survey.csv) scoring >= MIN_AUC.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import OOF_DIR, SUB_DIR, TARGET, folds, load

p = argparse.ArgumentParser()
p.add_argument("names", nargs="+")
p.add_argument("--out", default="stack")
p.add_argument("--C", type=float, default=1.0)
p.add_argument("--public", type=float, default=None)
p.add_argument("--exclude", default=None, help="regex of public names to drop")
args = p.parse_args()

train, test, _ = load()
y = train[TARGET].values


def logit(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def score(path):
    a = np.load(path, allow_pickle=False).astype("float64").ravel()
    return a if a.min() < 0 or a.max() > 1 else logit(a)  # some public files are already logits


oof_paths = [OOF_DIR / f"{n}_oof.npy" for n in args.names]
test_paths = [OOF_DIR / f"{n}_test.npy" for n in args.names]
if args.public is not None:
    pub = pd.read_csv(OOF_DIR.parent / "data" / "public_oof" / "survey.csv")
    pub = pub[pub.auc >= args.public]
    if args.exclude:
        pub = pub[~pub.name.str.contains(args.exclude)]
    args.names += list(pub.name)
    oof_paths += list(pub.oof)
    test_paths += list(pub.test)
X = np.column_stack([score(f) for f in oof_paths])
X_te = np.column_stack([score(f) for f in test_paths])
mu, sd = X.mean(0), X.std(0)
X, X_te = (X - mu) / sd, (X_te - mu) / sd
for n, col in zip(args.names, X.T):
    print(f"{n:20s} {roc_auc_score(y, col):.5f}")

oof = np.zeros(len(y))
for tr_idx, va_idx in folds(y):
    m = LogisticRegression(C=args.C, max_iter=1000).fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = m.decision_function(X[va_idx])
print(f"stack CV AUC: {roc_auc_score(y, oof):.5f}")

m = LogisticRegression(C=args.C, max_iter=1000).fit(X, y)
print("coefs:", dict(zip(args.names, np.round(m.coef_[0], 3))))
pred = m.predict_proba(X_te)[:, 1]
np.save(OOF_DIR / f"{args.out}_oof.npy", oof)  # logits, reusable as a level-2 input
np.save(OOF_DIR / f"{args.out}_test.npy", m.decision_function(X_te))
pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(SUB_DIR / f"{args.out}.csv", index=False)
print("wrote", SUB_DIR / f"{args.out}.csv")
