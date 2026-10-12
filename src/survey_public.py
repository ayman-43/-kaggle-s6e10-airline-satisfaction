"""Score every public OOF file against our train labels and pair it with its test file.

Writes data/public_oof/survey.csv (name, oof path, test path, auc, value range).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import TARGET, load

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_oof"
train, test, _ = load()
y = train[TARGET].values
rows = []
seen = set()
for f in sorted(ROOT.rglob("oof_*.npy")):
    if f.parent.name.startswith("raw_"):
        continue
    tf = f.with_name("test_" + f.name[4:])
    if not tf.exists():
        tf = f.with_name("pred_" + f.name[4:])
    try:
        o = np.load(f, allow_pickle=False).astype("float64").ravel()
        t = np.load(tf, allow_pickle=False).astype("float64").ravel() if tf.exists() else None
    except Exception as e:
        print("skip", f.name, e); continue
    if len(o) != len(y) or t is None or len(t) != len(test) or not np.isfinite(o).all():
        print("skip", f.name, len(o), None if t is None else len(t)); continue
    key = np.round(o[:2000], 5).tobytes()  # same model shipped in several datasets
    if key in seen:
        continue
    seen.add(key)
    rows.append(dict(name=f.parent.name + "/" + f.stem[4:], oof=str(f), test=str(tf),
                     auc=roc_auc_score(y, o), lo=o.min(), hi=o.max()))
df = pd.DataFrame(rows).sort_values("auc", ascending=False)
df.to_csv(ROOT / "survey.csv", index=False)
print(len(df), "usable"); print(df[["name", "auc", "lo", "hi"]].head(25).to_string(index=False))
