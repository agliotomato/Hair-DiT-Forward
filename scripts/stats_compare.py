"""
run 간 per-image 지표 유의차 확정 — paired test.

per-image CSV(eval_metrics.py 출력)들을 stem으로 정렬해 두 run을 짝지어 비교한다.
같은 test set·같은 stem이므로 **paired** 검정이 적절하다.

per-image 지표 4종만 대상: sketch_lpips, edge_iou, lpips(GT), bnd_lpips.
(FID 3종은 분포 단위라 per-image 짝이 없음 → 여기서 제외.)

각 (pair × metric × scope[combined/braid/unbraid])에 대해:
  - n, mean_A, mean_B, mean_diff(=A−B)
  - 차이의 95% CI (paired: 1.96·SD_diff/√n)
  - paired t-test p (scipy.stats.ttest_rel)
  - Wilcoxon signed-rank p (비모수)
  - 방향(↑/↓) 고려한 유의 우열 판정 (α=0.05)

Usage:
  python scripts/stats_compare.py \
    --per-image eval_results/gtcolor_mcs1_per_image.csv \
                eval_results/gtcolor_mcs3_per_image.csv \
                eval_results/gtcolor_mcs6_per_image.csv \
    --pairs mcs1:mcs6 mcs1:mcs3 \
    --out eval_results/gtcolor_stats.md

  # --pairs 생략 시 baseline(기본 첫 run) vs 나머지 전부
  python scripts/stats_compare.py --per-image eval_results/gtcolor_mcs*_per_image.csv --baseline mcs1
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from scipy import stats

# (key, label, higher_better)
METRICS = [
    ("sketch_lpips", "Sketch LPIPS", False),
    ("edge_iou",     "Edge IoU",     True),
    ("lpips",        "LPIPS (GT)",   False),
    ("bnd_lpips",    "Boundary LPIPS", False),
]
SCOPES = ["combined", "braid", "unbraid"]


def infer_name(path: Path) -> str:
    s = path.stem  # e.g. gtcolor_mcs1_per_image
    s = re.sub(r"_per_image$", "", s)
    s = re.sub(r"^(gtcolor|recheck|combined)_", "", s)
    return s


def load_csv(path: Path) -> dict[str, dict]:
    """stem → {metric: float|nan, split: str}"""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            stem = row["stem"]
            split = row.get("split") or ("braid" if stem.startswith("braid_") else "unbraid")
            rec = {"split": split}
            for key, *_ in METRICS:
                v = row.get(key, "")
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    fv = float("nan")
                rec[key] = fv
            out[stem] = rec
    return out


def paired_arrays(A: dict, B: dict, key: str, scope: str):
    """두 run에서 공통 stem·유효값만 정렬한 (a, b) 리스트."""
    a_vals, b_vals = [], []
    for stem in sorted(A.keys() & B.keys()):
        ra, rb = A[stem], B[stem]
        if scope != "combined" and ra["split"] != scope:
            continue
        va, vb = ra[key], rb[key]
        if math.isnan(va) or math.isnan(vb):
            continue
        a_vals.append(va); b_vals.append(vb)
    return a_vals, b_vals


def compare(a_vals, b_vals, higher_better: bool) -> dict | None:
    n = len(a_vals)
    if n < 2:
        return None
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    mean_a = sum(a_vals) / n
    mean_b = sum(b_vals) / n
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(n) if sd > 0 else 0.0

    # paired t-test
    try:
        t_p = float(stats.ttest_rel(a_vals, b_vals).pvalue)
    except Exception:
        t_p = float("nan")
    # Wilcoxon (차이가 전부 0이면 정의 불가)
    if all(d == 0 for d in diffs):
        w_p = 1.0
    else:
        try:
            w_p = float(stats.wilcoxon(a_vals, b_vals).pvalue)
        except Exception:
            w_p = float("nan")

    # 방향 고려 우열: higher_better면 A가 크면 A 우세
    a_better = (md > 0) if higher_better else (md < 0)
    sig = (not math.isnan(t_p) and t_p < 0.05) and (not math.isnan(w_p) and w_p < 0.05)
    if sig:
        verdict = ("A>B" if a_better else "B>A") + " (유의)"
    elif (not math.isnan(t_p) and t_p < 0.05) or (not math.isnan(w_p) and w_p < 0.05):
        verdict = ("A>B" if a_better else "B>A") + " (한쪽만)"
    else:
        verdict = "n.s."
    return {"n": n, "mean_a": mean_a, "mean_b": mean_b, "md": md, "ci": ci,
            "t_p": t_p, "w_p": w_p, "verdict": verdict}


def _p(p):
    if math.isnan(p):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def render(pairs, data, names):
    lines = []
    lines.append("# Run 간 per-image 유의차 (paired test)\n")
    lines.append("> paired t-test + Wilcoxon signed-rank. A−B 기준 mean_diff, 95% CI는 차이의 신뢰구간.")
    lines.append("> 유의 = 두 검정 모두 p<0.05. 'n.s.' = 유의차 없음.\n")
    for a_name, b_name in pairs:
        A, B = data[a_name], data[b_name]
        lines.append(f"\n## {a_name} (A) vs {b_name} (B)\n")
        lines.append("| Metric | scope | n | mean_A | mean_B | mean_diff(A−B) | 95% CI | t p | Wilcoxon p | 판정 |")
        lines.append("|---|---|--:|--:|--:|--:|--:|--:|--:|---|")
        for key, label, hb in METRICS:
            for scope in SCOPES:
                a_vals, b_vals = paired_arrays(A, B, key, scope)
                r = compare(a_vals, b_vals, hb)
                if r is None:
                    lines.append(f"| {label} | {scope} | {len(a_vals)} | — | — | — | — | — | — | n/a |")
                    continue
                lines.append(
                    f"| {label} | {scope} | {r['n']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
                    f"{r['md']:+.4f} | ±{r['ci']:.4f} | {_p(r['t_p'])} | {_p(r['w_p'])} | {r['verdict']} |"
                )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="run 간 per-image 지표 paired 유의차 검정")
    ap.add_argument("--per-image", nargs="+", required=True, help="per_image.csv 경로들")
    ap.add_argument("--names", nargs="+", default=None, help="run 이름(미지정 시 파일명에서 추론)")
    ap.add_argument("--pairs", nargs="+", default=None, help="A:B 쌍들 (예: mcs1:mcs6). 미지정 시 baseline vs 전부")
    ap.add_argument("--baseline", default=None, help="--pairs 미지정 시 기준 run (기본: 첫 run)")
    ap.add_argument("--out", default=None, help="markdown 출력 경로")
    args = ap.parse_args()

    paths = [Path(p) for p in args.per_image]
    names = args.names or [infer_name(p) for p in paths]
    if len(names) != len(paths):
        raise SystemExit("[ERROR] --names 개수가 --per-image 와 다릅니다.")
    data = {nm: load_csv(p) for nm, p in zip(names, paths)}

    if args.pairs:
        pairs = []
        for pr in args.pairs:
            a, b = pr.split(":")
            if a not in data or b not in data:
                raise SystemExit(f"[ERROR] 알 수 없는 run: {pr} (가능: {list(data)})")
            pairs.append((a, b))
    else:
        base = args.baseline or names[0]
        if base not in data:
            raise SystemExit(f"[ERROR] baseline '{base}' 없음 (가능: {list(data)})")
        pairs = [(base, b) for b in names if b != base]

    md = render(pairs, data, names)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
