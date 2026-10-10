"""RealMLP (pytabkit) with 5-fold CV. Every raw numeric column also gets a categorical twin,
so the net learns an embedding per distinct value (e.g. per route).

Usage: python src/train_nn.py [--v2|--v3] [--aux ev|all|allog|ogev] [--sched flat_anneal]
                             [--seed 0] [--epochs 3] [--name NAME]
"""
import argparse
import logging
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

from common import (BASE_FEATURES, CAT_COLS, N_FOLDS, NUM_COLS, RATING_COLS, TARGET,
                    add_features, aux_features, folds, load, save, v2_features)

warnings.filterwarnings("ignore")
for n in ("lightning", "lightning.pytorch", "pytorch_lightning"):
    logging.getLogger(n).setLevel(logging.ERROR)

p = argparse.ArgumentParser()
p.add_argument("--v2", action="store_true", help="add og logit, route profiles, counts, in-fold TE")
p.add_argument("--aux", choices=["ev", "all", "allog", "ogev"],
               help="append auxiliary-task features (data/aux.parquet)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--sched", default="lin_cos_log_15", help="lr schedule, e.g. flat_anneal")
p.add_argument("--v3", action="store_true", help="v2 with og_xgb and the wider TE keys")
p.add_argument("--name")
args = p.parse_args()

import torch
from pytabkit import RealMLP_TD_Classifier

# Hyper-parameters from Vladimir Demidov's public notebook "PS|S6|E10: RealMLP - PyTabKit".
REALMLP = dict(
    n_ens=8, n_epochs=args.epochs, batch_size=256, use_early_stopping=False,
    lr=0.053, wd=0.0236, sq_mom=0.988, lr_sched=args.sched, wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18,
    hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="expm4t",
    plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
    ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False,
    bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
)

train, test, orig = load()
train, test, orig = add_features(train), add_features(test), add_features(orig)
y = train[TARGET].values
X, X_test = train[BASE_FEATURES].copy(), test[BASE_FEATURES].copy()
te_cols = []
if args.v3:
    args.v2 = True
if args.v2:
    ex_tr, ex_te, keys_tr, keys_te = v2_features(train, test, orig, v3=args.v3)
    X, X_test = pd.concat([X, ex_tr], axis=1), pd.concat([X_test, ex_te], axis=1)
    te_cols = list(keys_tr.columns)
if args.aux:
    ax_tr, ax_te = aux_features(args.aux, len(X))
    X, X_test = pd.concat([X, ax_tr], axis=1), pd.concat([X_test, ax_te], axis=1)

cat_cols = list(CAT_COLS)
for df in (X, X_test):
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0.0)
for c in NUM_COLS + RATING_COLS:
    vals = sorted(set(X[c].astype(int).astype(str)) | set(X_test[c].astype(int).astype(str)))
    for df in (X, X_test):
        df[c + "_cat_"] = pd.Categorical(df[c].astype(int).astype(str), categories=vals)
    cat_cols.append(c + "_cat_")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| features:", X.shape[1] + len(te_cols), flush=True)

oof, test_pred, t0 = np.zeros(len(X)), np.zeros(len(X_test)), time.time()
for k, (tr_idx, va_idx) in enumerate(folds(y)):
    Xa, Xb, Xt = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy(), X_test.copy()
    if te_cols:
        enc = TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True,
                            random_state=0)
        Xa[te_cols] = enc.fit_transform(keys_tr.iloc[tr_idx], y[tr_idx])
        Xb[te_cols] = enc.transform(keys_tr.iloc[va_idx])
        Xt[te_cols] = enc.transform(keys_te)
    m = RealMLP_TD_Classifier(**REALMLP, device=device, random_state=args.seed, verbosity=0,
                              val_metric_name="1-auc_ovr")
    m.fit(Xa, y[tr_idx], X_val=Xb, y_val=y[va_idx], cat_col_names=cat_cols)
    oof[va_idx] = m.predict_proba(Xb)[:, 1]
    test_pred += m.predict_proba(Xt)[:, 1] / N_FOLDS
    print(f"fold {k}: auc={roc_auc_score(y[va_idx], oof[va_idx]):.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del m
    torch.cuda.empty_cache()

name = args.name or f"realmlp{'_v2' if args.v2 else ''}_s{args.seed}"
print(f"{name} CV AUC: {roc_auc_score(y, oof):.5f}")
save(name, oof, test_pred, test["id"].values)
