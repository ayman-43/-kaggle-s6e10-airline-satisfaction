#!/bin/sh
# Round 3: more seeds of the strongest round-2 models (runs after queue_round2.sh finishes).
cd "$(dirname "$0")/.."
while ! grep -q "CV AUC\|Traceback" logs/lgb_v3_ogev.log 2>/dev/null; do sleep 30; done
run() { name=$1; shift; echo "$(date +%T) start $name"; "$@" --name "$name" > "logs/$name.log" 2>&1; echo "$(date +%T) $(grep 'CV AUC' "logs/$name.log" || echo "$name FAILED")"; }
run realmlp_v3fa_aux_s2  python src/train_nn.py --v3 --aux all --sched flat_anneal --epochs 3 --seed 12
run realmlp_v3fa_aux_s3  python src/train_nn.py --v3 --aux allog --sched flat_anneal --epochs 4 --seed 13
run xgb_v3_aux_s2        python src/train.py xgb --v3 --aux all --seed 2
run lgb_v3_aux_s2        python src/train.py lgb --v3 --aux allog --seed 2
echo "$(date +%T) round3 done"
