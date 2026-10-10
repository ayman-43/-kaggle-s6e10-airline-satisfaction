"""Blend saved models by optimizing rank-average weights on OOF AUC.

Usage: python src/blend.py name1 name2 ... [--out blend]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import OOF_DIR, SUB_DIR, TARGET, load

p = argparse.ArgumentParser()
p.add_argument("names", nargs="+")
p.add_argument("--out", default="blend")
args = p.parse_args()

train, test, _ = load()
y = train[TARGET].values


def ranks(a):
    return rankdata(a) / len(a)


oofs = np.column_stack([ranks(np.load(OOF_DIR / f"{n}_oof.npy")) for n in args.names])
tests = np.column_stack([ranks(np.load(OOF_DIR / f"{n}_test.npy")) for n in args.names])
for n, col in zip(args.names, oofs.T):
    print(f"{n:20s} {roc_auc_score(y, col):.5f}")


def neg_auc(w):
    w = np.abs(w) / np.abs(w).sum()
    return -roc_auc_score(y, oofs @ w)


k = len(args.names)
res = minimize(neg_auc, np.ones(k) / k, method="Nelder-Mead", options={"maxiter": 500})
w = np.abs(res.x) / np.abs(res.x).sum()
print("weights:", dict(zip(args.names, np.round(w, 3))))
print(f"equal-weight AUC: {-neg_auc(np.ones(k)):.5f}   optimized AUC: {-res.fun:.5f}")

pd.DataFrame({"id": test["id"], TARGET: tests @ w}).to_csv(SUB_DIR / f"{args.out}.csv", index=False)
print("wrote", SUB_DIR / f"{args.out}.csv")
