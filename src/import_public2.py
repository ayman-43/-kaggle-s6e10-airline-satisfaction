"""Convert the second batch of public OOF datasets (data/public_oof/raw_*) into
oof_<name>.npy / test_<name>.npy pairs under data/public_oof/p2_<source>/ so that
survey_public.py picks them up. Rows are assumed to be in train/test id order;
csv files carrying an id column are sorted by it first.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_oof"


def save(src, name, oof, test):
    out = ROOT / f"p2_{src}"
    out.mkdir(exist_ok=True)
    np.save(out / f"oof_{name}.npy", np.asarray(oof, dtype="float64").ravel())
    np.save(out / f"test_{name}.npy", np.asarray(test, dtype="float64").ravel())


def csv_pred(f):
    d = pd.read_csv(f)
    if "id" in d:
        d = d.sort_values("id")
    col = [c for c in d.columns if c not in ("id", "fold")][-1]
    return d[col].values


for raw in sorted(ROOT.glob("raw_*")):
    src = raw.name[4:]
    n = 0
    # npy pairs: oof_X.npy + test_X.npy
    for f in raw.glob("oof_*.npy"):
        t = f.with_name("test_" + f.name[4:])
        if t.exists():
            save(src, f.stem[4:], np.load(f), np.load(t)); n += 1
    # npz with oof/test keys
    for f in raw.glob("*.npz"):
        z = np.load(f)
        if "oof" in z and "test" in z:
            save(src, f.stem, z["oof"], z["test"]); n += 1
    # csv pairs: X_oof.csv + X_test.csv
    for f in raw.glob("*_oof.csv"):
        stem = f.name[:-8]
        t = raw / f"{stem}_test.csv"
        if not t.exists():
            t = raw / f"{stem.split('_')[0]}_submission.csv"
        if t.exists():
            save(src, stem, csv_pred(f), csv_pred(t)); n += 1
    print(src, n)
