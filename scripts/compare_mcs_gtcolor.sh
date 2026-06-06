#!/usr/bin/env bash
# mcs1~6 를 GT 헤어색 sketch로 전체 test set(braid 107 + unbraid 466 = 573) 재생성 후 평가.
#
#   - 입력 sketch를 SHS 방식으로 GT(img×matte) per-stroke 평균색 재착색
#     (infer_custom.py --recolor_from_gt). 결정론적이라 6개 run이 동일 sketch 입력.
#   - 출력은 face 포함 full 합성. 생성 후 eval_metrics.py --split combined 로 평가.
#   - 기존 results/(컬러 주입 sketch) 평가와 분리: 출력은 outputs/gtcolor/, 평가는 eval_results/gtcolor_*.
#
# 사용:
#   bash scripts/compare_mcs_gtcolor.sh                      # 6개 전부
#   PY=.venv/bin/python bash scripts/compare_mcs_gtcolor.sh  # 로컬 venv
#   bash scripts/compare_mcs_gtcolor.sh mcs1 mcs6            # 일부만
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-outputs/gtcolor}
EVAL=${EVAL:-eval_results}
STEPS=${STEPS:-20}

# ── run → "config:checkpoint" 매핑 (서버 경로에 맞게 여기만 수정) ──
declare -A RUN
RUN[mcs1]="configs/mcs1_phase2.yaml:checkpoints/mcs1_phase2/epoch_40.pth"
RUN[mcs2]="configs/mcs2_phase2.yaml:checkpoints/mcs2_phase2/epoch_40.pth"
RUN[mcs3]="configs/mcs3_phase2.yaml:checkpoints/mcs3_phase2/epoch_40.pth"
RUN[mcs4]="configs/mcs4_phase2.yaml:checkpoints/mcs4_phase2/epoch_40.pth"
RUN[mcs5]="configs/mcs5_phase2.yaml:checkpoints/mcs5_phase2/epoch_40.pth"
RUN[mcs6]="configs/mcs6_phase2.yaml:checkpoints/mcs6_phase2/epoch_40.pth"
ORDER=(mcs1 mcs2 mcs3 mcs4 mcs5 mcs6)

# 인자 주면 그 run만
SELECT=("$@")
selected() { [ ${#SELECT[@]} -eq 0 ] && return 0; for s in "${SELECT[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

mkdir -p "$OUT" "$EVAL"

infer_set() {  # $1=run $2=set(braid|unbraid) $3=config $4=ckpt
  local name=$1 set=$2 cfg=$3 ckpt=$4
  local dst="$OUT/${name}_${set}"
  echo "---- $name / $set → $dst ----"
  $PY scripts/infer_custom.py \
    --sketch "dataset/$set/sketch/test" \
    --matte  "dataset/$set/matte/test" \
    --face   "dataset/$set/img/test" \
    --recolor_from_gt \
    --checkpoint "$ckpt" --config "$cfg" \
    --num_steps "$STEPS" \
    --output_dir "$dst"
}

for name in "${ORDER[@]}"; do
  selected "$name" || continue
  IFS=':' read -r cfg ckpt <<< "${RUN[$name]}"
  if [ ! -f "$ckpt" ]; then echo "[SKIP] $name — checkpoint 없음: $ckpt"; continue; fi
  echo "########## $name ##########"
  infer_set "$name" braid   "$cfg" "$ckpt"
  infer_set "$name" unbraid "$cfg" "$ckpt"

  echo "---- $name 평가 (combined, N=573) ----"
  $PY scripts/eval_metrics.py \
    --split combined \
    --pred  "$OUT/${name}_braid" "$OUT/${name}_unbraid" \
    --tag   "$name" \
    --out   "$EVAL/gtcolor_${name}"
done

echo "===== GT-color 평가 요약 (7지표) ====="
$PY - "$EVAL" "${SELECT[@]:-}" <<'PY'
import csv, glob, os, sys
evaldir = sys.argv[1]; sel = [a for a in sys.argv[2:] if a]
files = sorted(glob.glob(os.path.join(evaldir, "gtcolor_*_summary.csv")))
cols, data = [], {}
for f in files:
    name = os.path.basename(f)[len("gtcolor_"):-len("_summary.csv")]
    if sel and name not in sel: continue
    cols.append(name)
    with open(f) as fh:
        data[name] = {row["metric"]: row for row in csv.DictReader(fh)}
if not cols:
    print("(summary 없음)"); raise SystemExit

# (label, unit) — split=macro 보고, combined=통합573 보고
SPEC = [("Sketch LPIPS ↓","split"),("Edge IoU ↑","split"),("Hair FID ↓","combined"),
        ("LPIPS (GT) ↓","split"),("Boundary FID ↓","combined"),("Boundary LPIPS ↓","split"),
        ("Full-portrait FID ↓","combined")]

def cell(run, label, unit):
    r = data[run].get(label)
    if r is None: return "?"
    return r["combined573"] if unit == "combined" else r["macro"]

print("\n# 보고값 (per-image=macro, FID=통합573)")
print("| Metric | 단위 | " + " | ".join(cols) + " |")
print("|" + "---|" * (len(cols) + 2))
for label, unit in SPEC:
    tag = "573" if unit == "combined" else "macro"
    print(f"| {label} | {tag} | " + " | ".join(cell(c, label, unit) for c in cols) + " |")

print("\n# per-image 도메인 분해 (braid ±CI95 / unbraid)")
print("| Metric | " + " | ".join(f"{c}(braid)" for c in cols) + " | " + " | ".join(f"{c}(unbr)" for c in cols) + " |")
print("|" + "---|" * (2 * len(cols) + 1))
for label, unit in SPEC:
    if unit != "split": continue
    braids = []
    for c in cols:
        r = data[c].get(label, {})
        v = r.get("braid", "?"); ci = r.get("braid_ci95", "")
        braids.append(f"{v}±{ci}" if ci not in ("", "N/A") else v)
    unbrs = [data[c].get(label, {}).get("unbraid", "?") for c in cols]
    print(f"| {label} | " + " | ".join(braids) + " | " + " | ".join(unbrs) + " |")
PY

echo "===== 유의차 검정 (paired t-test + Wilcoxon, baseline=mcs1) ====="
csvs=$(ls "$EVAL"/gtcolor_*_per_image.csv 2>/dev/null || true)
if [ -n "$csvs" ] && [ -f "$EVAL/gtcolor_mcs1_per_image.csv" ]; then
  $PY scripts/stats_compare.py --per-image $csvs \
    --baseline mcs1 --out "$EVAL/gtcolor_stats.md" || true
elif [ -n "$csvs" ]; then
  $PY scripts/stats_compare.py --per-image $csvs --out "$EVAL/gtcolor_stats.md" || true
fi

echo "===== 완료 ====="
echo "  생성:     $OUT/"
echo "  평가:     $EVAL/gtcolor_*_summary.csv  (braid/unbraid/macro ±CI95, FID 통합573)"
echo "  per-img:  $EVAL/gtcolor_*_per_image.csv"
echo "  유의차:   $EVAL/gtcolor_stats.md  (mcs1 vs 나머지 paired test)"
