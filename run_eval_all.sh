#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# exp1_braid는 이미 완료



# ── Exp4 ─────────────────────────────────────────────────────
python scripts/infer_custom.py \
  --sketch     dataset/braid/sketch/test \
  --matte      dataset/braid/matte/test \
  --face       dataset/braid/img/test \
  --checkpoint checkpoints/exp4_phase2/final.pth \
  --config     configs/exp4_phase2.yaml \
  --output_dir outputs/exp4_braid

python scripts/infer_custom.py \
  --sketch     dataset/unbraid/sketch/test \
  --matte      dataset/unbraid/matte/test \
  --face       dataset/unbraid/img/test \
  --checkpoint checkpoints/exp4_phase2/final.pth \
  --config     configs/exp4_phase2.yaml \
  --output_dir outputs/exp4_unbraid

python scripts/eval_metrics.py \
  --split combined \
  --pred  outputs/exp4_braid outputs/exp4_unbraid \
  --pred-suffix _full \
  --tag "Exp4"

echo "===== 전체 완료 ====="
