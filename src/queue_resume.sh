#!/bin/sh
# Resume after shutdown: unfinished round-2 models + round-3 seeds. Trees first (lighter on RAM), slow CatBoost last.
cd "$(dirname "$0")/.."
run() { name=$1; shift; echo "$(date +%T) start $name"; "$@" --name "$name" > "logs/$name.log" 2>&1; echo "$(date +%T) $(grep 'CV AUC' "logs/$name.log" || echo "$name FAILED")"; }
run lgb_v3_ogev          python src/train.py lgb --v3 --aux ogev --seed 7
run xgb_v3_aux_s2        python src/train.py xgb --v3 --aux all --seed 2
run lgb_v3_aux_s2        python src/train.py lgb --v3 --aux allog --seed 2
run realmlp_v3fa_aux_s2  python src/train_nn.py --v3 --aux all --sched flat_anneal --epochs 3 --seed 12
run realmlp_v3fa_aux_s3  python src/train_nn.py --v3 --aux allog --sched flat_anneal --epochs 4 --seed 13
run cat_ctr              python src/train.py cat --v3 --no-te --l2 10 --depth 7
echo "$(date +%T) resume queue done"
