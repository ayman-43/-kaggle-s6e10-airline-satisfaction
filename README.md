# Kaggle Playground S6E10: Predicting Airline Passenger Satisfaction

My solution to [Kaggle Playground Series, Season 6 Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10).
The task is to predict the probability that an airline passenger is **satisfied**. Submissions are scored with **ROC AUC**.

**Current best:** public LB **0.96106** (rank 414 of 1,414 on 2026-10-10), 5-fold CV **0.96153**.

The competition data is synthetic, generated from the real
[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) dataset
(about 130k rows), which this project also uses as external data.

---

## Contents

```
src/
  common.py          data loading, feature engineering, fold split, saving predictions
  train.py           gradient-boosted trees (LightGBM / XGBoost / CatBoost), 5-fold CV
  train_nn.py        RealMLP neural net (pytabkit), 5-fold CV
  aux_features.py    auxiliary-task ("expected rating") features -> data/aux.parquet
  stack.py           logistic-regression stack over saved out-of-fold predictions
  blend.py           alternative: rank-average blend with optimised weights
  queue_round2.sh    the round-2 training queue (runs the models one after another)
notebooks/
  s6e10_pipeline.ipynb   the full pipeline in one notebook, with EDA (runs locally or on Kaggle)
logs/                training logs with per-fold and overall CV AUC for every model
```

Generated, not committed: `data/` (input data), `oof/` (out-of-fold and test predictions, `.npy`),
`submissions/` (submission CSVs) and `catboost_info/`.

## Setup

```bash
pip install -r requirements.txt

# Kaggle API credentials must be configured (https://www.kaggle.com/docs/api)
kaggle competitions download -c playground-series-s6e10 -p data && unzip data/playground-series-s6e10.zip -d data
kaggle datasets download -d teejmahal20/airline-passenger-satisfaction -p data/original --unzip
```

Expected layout: `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`,
`data/original/train.csv`, `data/original/test.csv`.

A CUDA GPU is assumed for XGBoost, CatBoost and RealMLP. All of it ran on a 4 GB RTX 2050.

## How to run

Every script uses the same split (`StratifiedKFold(5, shuffle=True, random_state=42)`), so out-of-fold
predictions line up row for row and can be stacked. Each run saves `oof/<name>_oof.npy`,
`oof/<name>_test.npy` and `submissions/<name>.csv`.

```bash
# trees
python src/train.py xgb --v3 --name xgb_v3
python src/train.py lgb --v3 --name lgb_v3
python src/train.py cat --v3 --name cat_v3

# neural net
python src/train_nn.py --v2 --seed 0 --name realmlp_v2_s0

# auxiliary-task features (once), then models that use them
python src/aux_features.py
python src/train.py xgb --v3 --aux all --name xgb_v3_aux
python src/train_nn.py --aux all --sched flat_anneal --epochs 4 --seed 23 --name realmlp_fa_aux

# stack any set of saved models
python src/stack.py lgb_v3 xgb_v3 xgb_v2_nocat realmlp_s0 cat_noorig --out stack_v7

# submit
kaggle competitions submit -c playground-series-s6e10 -f submissions/stack_v7.csv -m "stack_v7"
```

### Main options

| Option | Script | Meaning |
|---|---|---|
| `--v2` | both | adds route features, value counts, original-data model logit and in-fold target encoding |
| `--v3` | both | `--v2` + an XGBoost original-data model + wider target-encoding keys |
| `--aux ev\|all\|allog\|ogev` | both | appends auxiliary-task features (see below) |
| `--no-orig` | train.py | don't append the original dataset to the training folds (implied by `--v2`) |
| `--as-cat` | train.py | categorical copies of every numeric column (LightGBM/XGBoost) |
| `--no-te` | train.py | v2/v3 features without our target encoding (for CatBoost's own statistics) |
| `--l2`, `--depth` | train.py | CatBoost `l2_leaf_reg` and depth |
| `--max-bin` | train.py | XGBoost histogram bins |
| `--sched` | train_nn.py | RealMLP learning-rate schedule, e.g. `flat_anneal` |
| `--epochs` | train_nn.py | RealMLP epochs |
| `--seed`, `--name` | both | random seed and output name |

## Approach

### 1. Flight Distance works as a route ID
In the synthetic data, `Flight Distance` behaves less like a distance and more like an identifier for a
route. Several features exploit this:
- **Route profile:** for each Flight Distance value, the mean of every other column (`fdm_*`), plus the
  route's row count. No target is used.
- **Value counts** of each numeric column over train + test (`cnt_*`).
- **In-fold target encoding** of route keys such as route, route × class and route × travel type.
  Training rows are cross-fitted inside each fold, so no row ever sees its own label.
- **v3** adds route × each rating, passenger segment × each rating, and coarse route groups
  (`distance // 10`).

### 2. Knowledge from the original dataset
LightGBM and XGBoost models trained **only on the original dataset** predict every competition row, and
their logits (`og_lgb`, `og_xgb`) become features. No competition label is involved, so this is leak-free.
This works better than appending the original rows to the training data.

### 3. Auxiliary-task features (`aux_features.py`)
For each of the 13 ratings and 3 categorical columns, an XGBoost predicts that column from the other 20
(cross-fitted over train + test, no target). The features are the probability of the actual value, the
expected rating and a per-row log-likelihood. They capture how "typical" a passenger's answers are. A
second set comes from predictors trained only on the original data (`ogaux_*`).

### 4. Models
- **XGBoost / LightGBM:** depth 8 / 63 leaves, learning rate 0.02, early stopping on AUC.
- **CatBoost:** every numeric column is also passed as a categorical string, so CatBoost's ordered target
  statistics pick up synthetic-data value artifacts.
- **RealMLP** (pytabkit): every numeric column gets a categorical twin with a learned embedding. Optional
  `flat_anneal` learning-rate schedule.

### 5. Stacking
A logistic regression on the out-of-fold **logits** of every model, scored with the same 5 folds.
Models are chosen by CV, not by public leaderboard score.

## Results

5-fold CV AUC (same folds throughout):

| Model | CV AUC |
|---|---|
| LightGBM, raw features | 0.95882 |
| XGBoost, raw features | 0.95865 |
| LightGBM, + numeric columns as categories | 0.96002 |
| CatBoost, numeric columns as categories | 0.96056 |
| RealMLP, raw + categorical twins | 0.96075 |
| CatBoost v2 | 0.96082 |
| CatBoost v3 | 0.96100 |
| LightGBM v2 | 0.96105 |
| RealMLP v2 | 0.96107 |
| XGBoost v2 | 0.96108 |
| XGBoost v3 | 0.96127 |
| **LightGBM v3** | **0.96130** |

Submissions:

| Submission | Models | CV AUC | Public LB |
|---|---|---|---|
| stack_v2 | XGBoost/LightGBM v2 + CatBoost | 0.96136 | 0.96102 |
| stack_v5 | 12 models (v2 trees + RealMLP, several seeds) | 0.96144 | 0.96101 |
| **stack_v7** | 10 models (LightGBM v3 + XGBoost v3 + XGBoost v2 seeds, CatBoost, RealMLP) | **0.96153** | **0.96106** |

Round 2 (auxiliary-task features, `flat_anneal` RealMLPs, CatBoost with its own categorical statistics) is
in progress. See `src/queue_round2.sh` and `logs/`.

## What didn't help
- Appending the original dataset to the training folds once the original-model logit was a feature.
- `--as-cat` categorical copies for XGBoost on top of the v2 features (0.96037 vs 0.96108 without).
- Adding CatBoost v3 to the stack: its CV was unchanged (0.9615332 vs 0.9615343 without it).

## Credits
- goodpjw2008's public S6E10 notebook series (route-ID features, auxiliary-task features, `flat_anneal`
  RealMLP and the technique-by-technique scoreboard), which guided much of round 2.
- Vladimir Demidov's public notebook "PS|S6|E10: RealMLP - PyTabKit" for the RealMLP hyper-parameters.
- [pytabkit](https://github.com/dholzmueller/pytabkit) for RealMLP.
