#!/bin/sh
# Round 2: auxiliary-task features, flat_anneal RealMLPs, CatBoost with its own categorical statistics.
cd "$(dirname "$0")/.."
while [ ! -f data/aux.parquet ]; do sleep 20; done
run() { name=$1; shift; echo "$(date +%T) start $name"; "$@" --name "$name" > "logs/$name.log" 2>&1; echo "$(date +%T) $(grep 'CV AUC' "logs/$name.log" || echo "$name FAILED")"; }
run xgb_v3_aux        python src/train.py xgb --v3 --aux all
run lgb_v3_aux        python src/train.py lgb --v3 --aux all
run realmlp_fa_aux    python src/train_nn.py --aux all --sched flat_anneal --epochs 4 --seed 23
run realmlp_v3fa_aux  python src/train_nn.py --v3 --aux all --sched flat_anneal --epochs 3 --seed 11
run realmlp_fa        python src/train_nn.py --sched flat_anneal --epochs 4 --seed 42
run xgb_v3_auxog      python src/train.py xgb --v3 --aux allog --seed 1
run realmlp_fa_auxog  python src/train_nn.py --aux allog --sched flat_anneal --epochs 4 --seed 32
run cat_ctr           python src/train.py cat --v3 --no-te --l2 10 --depth 7
run lgb_v3_ogev       python src/train.py lgb --v3 --aux ogev --seed 7
echo "$(date +%T) queue done"
