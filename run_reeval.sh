#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

for exp in 1 2 3 4; do
  echo "===== Exp${exp} 재평가 ====="
  python scripts/eval_metrics.py \
    --split combined \
    --pred  outputs/exp${exp}_braid outputs/exp${exp}_unbraid \
    --pred-suffix _full \
    --tag "Exp${exp}"
done

echo "===== 전체 완료 ====="
