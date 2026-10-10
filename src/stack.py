"""Stack saved models with a logistic regression on their OOF logits (scored with the same 5 folds).

Usage: python src/stack.py name1 name2 ... [--out stack] [--C 1.0]
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
args = p.parse_args()

train, test, _ = load()
y = train[TARGET].values


def logit(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


X = np.column_stack([logit(np.load(OOF_DIR / f"{n}_oof.npy")) for n in args.names])
X_te = np.column_stack([logit(np.load(OOF_DIR / f"{n}_test.npy")) for n in args.names])
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
pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(SUB_DIR / f"{args.out}.csv", index=False)
print("wrote", SUB_DIR / f"{args.out}.csv")
