"""
Schedule Ablation 정량 평가: none / back_only / all / front_only

각 schedule의 outputs/{schedule}/{stem}_hairpatch.png 를 GT와 비교.

Usage:
  python scripts/eval_schedule_ablation.py
  python scripts/eval_schedule_ablation.py --outputs_root /path/to/outputs

Output:
  eval_results/schedule_ablation_summary.csv
  eval_results/schedule_ablation_per_image.csv
  콘솔: 그룹별 비교표 + LaTeX
"""

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))
import eval_all as M  # noqa: E402

ROOT        = Path(__file__).parent.parent
GT_IMG_DIR  = ROOT / "dataset/braid/img/test"
GT_MATTE_DIR= ROOT / "dataset/braid/matte/test"
SKETCH_DIR  = ROOT / "dataset/braid/sketch/test"
OUT_DIR     = ROOT / "eval_results"

SCHEDULES   = ["none", "back_only", "all", "front_only"]

METRIC_GROUPS = [
    ("Sketch Fidelity", [
        ("Edge IoU ↑",       "edge_iou",     None,       True),
        ("Chamfer Dist ↓",   "chamfer",      None,       False),
        ("Sketch LPIPS ↓",   "sketch_lpips", None,       False),
    ]),
    ("Generation Quality", [
        ("Hair FID ↓",       None,           "hair_fid", False),
        ("LPIPS (GT) ↓",     "lpips",        None,       False),
        ("SSIM (GT) ↑",      "ssim",         None,       True),
        ("PSNR (GT) ↑",      "psnr",         None,       True),
    ]),
    ("Boundary Quality", [
        ("Boundary FID ↓",   None,           "bnd_fid",  False),
        ("Boundary LPIPS ↓", "bnd_lpips",    None,       False),
    ]),
    ("Identity", [
        ("Face LPIPS ↓",     "face_lpips",   None,       False),
        ("ArcFace Cos ↑",    "arcface_cos",  None,       True),
    ]),
]

PER_IMG_KEYS = [
    "edge_iou", "chamfer", "sketch_lpips",
    "psnr", "ssim", "lpips",
    "bnd_lpips", "face_lpips", "arcface_cos",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: Path, hw=None) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    if hw and img.shape[:2] != tuple(hw):
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return img


def safe_mean(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vs) / len(vs) if vs else None


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(outputs_root: Path):
    OUT_DIR.mkdir(exist_ok=True)

    stems = sorted(p.stem for p in GT_IMG_DIR.glob("*.png")
                   if (GT_MATTE_DIR / f"{p.stem}.png").exists()
                   and (SKETCH_DIR / f"{p.stem}.png").exists())

    # 각 schedule에 대해 출력 파일이 있는 stem만 교집합
    for sched in SCHEDULES:
        sdir = outputs_root / sched
        available = {p.stem[:-len("_hairpatch")] for p in sdir.glob("*_hairpatch.png")
                     if p.stem.endswith("_hairpatch")}
        stems = [s for s in stems if s in available]

    print(f"평가 대상 stem: {len(stems)}개")
    if len(stems) < 50:
        print("  [WARNING] FID는 샘플 수가 적으면 부정확합니다. 참고용으로만 사용하세요.")

    fid_bufs = {
        sched: {"hair_real": [], "hair_fake": [], "bnd_real": [], "bnd_fake": []}
        for sched in SCHEDULES
    }
    rows = []

    for stem in tqdm(stems, desc="이미지 평가"):
        gt_img = load_image(GT_IMG_DIR / f"{stem}.png")
        matte  = np.array(Image.open(GT_MATTE_DIR / f"{stem}.png").convert("L"))
        sketch = load_image(SKETCH_DIR / f"{stem}.png")
        hw     = gt_img.shape[:2]

        hair_mask = matte > 127
        bnd_mask  = M.get_boundary_mask(matte)
        sk_edge   = M.canny_edges(sketch)

        gt_hair_crop = M.extract_region_crop(gt_img, hair_mask)
        gt_bnd_crop  = M.extract_region_crop(gt_img, bnd_mask, min_px=16)

        row = {"stem": stem}

        for sched in SCHEDULES:
            pred_path = outputs_root / sched / f"{stem}_hairpatch.png"
            pred = load_image(pred_path, hw=hw)
            pred_edge = M.canny_edges(pred)

            row[f"{sched}_edge_iou"]     = M.edge_iou(pred_edge, sk_edge, matte)
            row[f"{sched}_chamfer"]      = M.chamfer_distance(pred_edge, sk_edge, matte)
            row[f"{sched}_sketch_lpips"] = M.sketch_lpips(pred, sketch, matte)

            gq = M.gen_quality_metrics(pred, gt_img, matte)
            row[f"{sched}_psnr"]  = gq["psnr"]
            row[f"{sched}_ssim"]  = gq["ssim"]
            row[f"{sched}_lpips"] = gq["lpips"]

            row[f"{sched}_bnd_lpips"]   = M.boundary_lpips(pred, gt_img, matte)
            row[f"{sched}_face_lpips"]  = M.face_lpips(pred, gt_img, matte)
            row[f"{sched}_arcface_cos"] = M.arcface_cosine(pred, gt_img, matte)

            pred_hair_crop = M.extract_region_crop(pred, hair_mask)
            pred_bnd_crop  = M.extract_region_crop(pred, bnd_mask, min_px=16)

            if gt_hair_crop is not None and pred_hair_crop is not None:
                fid_bufs[sched]["hair_real"].append(gt_hair_crop)
                fid_bufs[sched]["hair_fake"].append(pred_hair_crop)
            if gt_bnd_crop is not None and pred_bnd_crop is not None:
                fid_bufs[sched]["bnd_real"].append(gt_bnd_crop)
                fid_bufs[sched]["bnd_fake"].append(pred_bnd_crop)

        rows.append(row)

    # FID
    print("\nFID 계산 중...")
    fid_results = {}
    for sched in SCHEDULES:
        fid_results[f"{sched}_hair_fid"] = M.compute_fid(
            fid_bufs[sched]["hair_real"], fid_bufs[sched]["hair_fake"])
        fid_results[f"{sched}_bnd_fid"] = M.compute_fid(
            fid_bufs[sched]["bnd_real"], fid_bufs[sched]["bnd_fake"])
        print(f"  {sched}: Hair FID={fmt(fid_results[f'{sched}_hair_fid'])}  "
              f"Bnd FID={fmt(fid_results[f'{sched}_bnd_fid'])}")

    # Summary 집계
    summary_rows = []
    for _, metrics in METRIC_GROUPS:
        for label, per_key, fid_key, _ in metrics:
            vals = []
            for sched in SCHEDULES:
                if fid_key:
                    v = fid_results.get(f"{sched}_{fid_key}")
                else:
                    v = safe_mean([r.get(f"{sched}_{per_key}") for r in rows])
                vals.append(v)
            summary_rows.append([label] + [fmt(v) for v in vals])

    # CSV 저장
    with open(OUT_DIR / "schedule_ablation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + SCHEDULES)
        w.writerows(summary_rows)

    fieldnames = ["stem"] + [f"{s}_{k}" for s in SCHEDULES for k in PER_IMG_KEYS]
    with open(OUT_DIR / "schedule_ablation_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 콘솔 출력
    col_w = 12
    header = f"{'Metric':<22}" + "".join(f"  {s:>{col_w}}" for s in SCHEDULES)
    sep    = "=" * (22 + (col_w + 2) * len(SCHEDULES))
    print(f"\n{sep}")
    print(f"  Schedule Ablation — n={len(stems)}")
    print(sep)
    print(header)
    print(sep)

    row_idx = 0
    for group_name, metrics in METRIC_GROUPS:
        print(f"\n--- {group_name} ---")
        for label, _, _, higher_is_better in metrics:
            srow   = summary_rows[row_idx]; row_idx += 1
            strs   = srow[1:]
            try:
                floats = [float(s) for s in strs]
                best   = max(floats) if higher_is_better else min(floats)
                disp   = [f"[{s}]" if abs(float(s) - best) < 1e-9 else s for s in strs]
            except ValueError:
                disp = strs
            print(f"  {label:<20}" + "".join(f"  {d:>{col_w}}" for d in disp))
    print(sep)

    # LaTeX
    col_labels = ["none (base)", "back\\_only (Ours)", "all", "front\\_only"]
    col_spec   = "l" + "c" * len(SCHEDULES)
    print(f"\n% LaTeX — Schedule Ablation")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\hline")
    print("Metric & " + " & ".join(col_labels) + " \\\\")
    print("\\hline")

    row_idx = 0
    for group_name, metrics in METRIC_GROUPS:
        print(f"\\multicolumn{{{len(SCHEDULES)+1}}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        for label, _, _, higher_is_better in metrics:
            srow   = summary_rows[row_idx]; row_idx += 1
            strs   = srow[1:]
            try:
                floats = [float(s) for s in strs]
                best   = max(floats) if higher_is_better else min(floats)
                tex    = [f"\\textbf{{{s}}}" if abs(float(s) - best) < 1e-9 else s
                          for s in strs]
            except ValueError:
                tex = strs
            clean = label.replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$")
            print(f"  {clean} & " + " & ".join(tex) + " \\\\")

    print("\\hline")
    print("\\end{tabular}")

    print(f"\n저장: eval_results/schedule_ablation_summary.csv")
    print(f"      eval_results/schedule_ablation_per_image.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs_root",
        default=str(ROOT / "outputs"),
        help="none/back_only/all/front_only 폴더가 있는 상위 디렉토리",
    )
    args = parser.parse_args()
    M.try_load_face_model()
    evaluate(Path(args.outputs_root))
