"""
Split-aware evaluation script.

Usage:
  python3 scripts/eval_metrics.py \\
      --split   braid                          \\
      --pred    custom_results/dit/exp1        \\
      [--pred-suffix _full]                    \\
      [--out    eval_results/exp1_braid]       \\
      [--tag    "Exp1"]

Dataset paths are derived automatically from the split:
  dataset/{split}/img/test    — GT images
  dataset/{split}/matte/test  — GT mattes (grayscale)
  dataset/{split}/sketch/test — sketch images

Metrics (최종 7개 — 스펙 확정):
  per-image (braid/unbraid/macro 분리 보고, braid는 ±CI95):
    ① Sketch LPIPS ↓  (구조,     paired vs sketch, hair 영역)
    ② Edge IoU ↑      (구조 보조, paired vs sketch, hair 영역)
    ④ LPIPS (GT) ↓    (외형,     paired vs GT,     hair-masked)
    ⑥ Boundary LPIPS ↓(경계 보조, paired vs GT,     경계 band)
  FID (통합 573만 보고, dims=2048):
    ③ Hair FID ↓          (리얼리즘,   hair 마스킹/크롭)
    ⑤ Boundary FID ↓      (경계,       경계 band)
    ⑦ Full-portrait FID ↓ (전역 조화,  whole-image)

  보고 단위 규칙:
    - FID(③⑤⑦)는 2048-dim 공분산 → n=107 braid 단독은 rank-deficient. 통합(573)만 보고.
    - per-image(①②④⑥)는 unbraid/braid/macro 분리. braid는 ±CI95.

Outputs:
  <out>_per_image.csv  (stem, split, ①②④⑥)
  <out>_summary.csv    (metric, axis, paired, unit, braid, braid_ci95, unbraid, macro, combined573)
  console table
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLITS = {
    "braid": {
        "img":    ROOT / "dataset/braid/img/test",
        "matte":  ROOT / "dataset/braid/matte/test",
        "sketch": ROOT / "dataset/braid/sketch/test",
    },
    "unbraid": {
        "img":    ROOT / "dataset/unbraid/img/test",
        "matte":  ROOT / "dataset/unbraid/matte/test",
        "sketch": ROOT / "dataset/unbraid/sketch/test",
    },
    "combined": None,  # braid + unbraid 합산 — --pred braid_dir unbraid_dir 순서로 2개 지정
}

# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------
_lpips_fn = None


def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    return _lpips_fn


def _to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(DEVICE)


def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    fn = get_lpips()
    with torch.no_grad():
        ta, tb = _to_lpips_tensor(a), _to_lpips_tensor(b)
        if ta.shape[-1] < 64 or ta.shape[-2] < 64:
            ta = F.interpolate(ta, (64, 64), mode="bilinear", align_corners=False)
            tb = F.interpolate(tb, (64, 64), mode="bilinear", align_corners=False)
        return float(fn(ta, tb).item())


# ---------------------------------------------------------------------------
# Pixel metrics
# ---------------------------------------------------------------------------

def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * math.log10(255.0 ** 2 / err) if err > 0 else float("inf")


def _ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k2d = (lambda k: k @ k.T)(cv2.getGaussianKernel(11, 1.5))
    filt = lambda x: cv2.filter2D(x, -1, k2d)
    mu_a, mu_b = filt(a), filt(b)
    num = (2 * mu_a * mu_b + C1) * (2 * (filt(a * b) - mu_a * mu_b) + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (filt(a * a) - mu_a ** 2 + filt(b * b) - mu_b ** 2 + C2)
    return float((num / den).mean())


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([_ssim_channel(a[:, :, c], b[:, :, c]) for c in range(3)]))


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def _bbox_crop(img: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img.copy(), mask.copy()
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def extract_region_crop(img: np.ndarray, mask: np.ndarray, min_px: int = 64):
    if mask.sum() < min_px:
        return None
    crop, m = _bbox_crop(img, mask)
    crop[~m] = 0
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return cv2.resize(crop, (128, 128))


def get_boundary_mask(matte: np.ndarray) -> np.ndarray:
    return (matte >= 25) & (matte <= 230)


# ---------------------------------------------------------------------------
# [1] Sketch Fidelity
# ---------------------------------------------------------------------------

def canny_edges(img: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 50, 150) > 0


def edge_iou(pred_e: np.ndarray, sk_e: np.ndarray, matte: np.ndarray) -> float:
    hair = matte > 127
    a, b = pred_e & hair, sk_e & hair
    union = (a | b).sum()
    return float((a & b).sum() / union) if union > 0 else 0.0


def chamfer_distance(pred_e: np.ndarray, sk_e: np.ndarray, matte: np.ndarray) -> float:
    hair = matte > 127
    pts_a = np.argwhere(pred_e & hair).astype(np.float32)
    pts_b = np.argwhere(sk_e   & hair).astype(np.float32)
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("nan")
    d_ab, _ = cKDTree(pts_b).query(pts_a, k=1)
    d_ba, _ = cKDTree(pts_a).query(pts_b, k=1)
    return float((d_ab.mean() + d_ba.mean()) / 2)


def sketch_lpips(pred: np.ndarray, sketch: np.ndarray, matte: np.ndarray) -> float:
    hair = matte > 127
    edge_rgb = np.stack([canny_edges(pred).astype(np.uint8) * 255] * 3, axis=-1)
    p_crop, mc = _bbox_crop(edge_rgb, hair)
    s_crop, _  = _bbox_crop(sketch,   hair)
    p_crop[~mc] = 0
    s_crop[~mc] = 0
    if p_crop.shape[0] < 8 or p_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(p_crop, s_crop)


# ---------------------------------------------------------------------------
# [2] Generation Quality
# ---------------------------------------------------------------------------

def gen_quality_metrics(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> dict:
    hair = matte > 127
    pred_crop, mc = _bbox_crop(pred, hair)
    gt_crop,   _  = _bbox_crop(gt,   hair)
    pred_crop[~mc] = 0
    gt_crop  [~mc] = 0
    lpips_val = (compute_lpips(pred_crop, gt_crop)
                 if pred_crop.shape[0] >= 8 and pred_crop.shape[1] >= 8
                 else float("nan"))
    return {
        "psnr":  compute_psnr(pred[hair], gt[hair]),
        "ssim":  compute_ssim(pred_crop, gt_crop),
        "lpips": lpips_val,
    }


# ---------------------------------------------------------------------------
# [3] Boundary Quality
# ---------------------------------------------------------------------------

def boundary_lpips(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    bnd = get_boundary_mask(matte)
    if bnd.sum() < 64:
        return float("nan")
    p_crop, bc = _bbox_crop(pred, bnd)
    g_crop, _  = _bbox_crop(gt,   bnd)
    p_crop[~bc] = 0
    g_crop[~bc] = 0
    if p_crop.shape[0] < 8 or p_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(p_crop, g_crop)


# ---------------------------------------------------------------------------
# [4] Identity
# ---------------------------------------------------------------------------

def face_lpips(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    face = matte < 127
    if face.sum() < 64:
        return float("nan")
    p_crop, fc = _bbox_crop(pred, face)
    g_crop, _  = _bbox_crop(gt,   face)
    p_crop[~fc] = 0
    g_crop[~fc] = 0
    if p_crop.shape[0] < 8 or p_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(p_crop, g_crop)


_face_model      = None
_face_model_name = "none"


def try_load_face_model():
    global _face_model, _face_model_name
    for loader in [_load_facenet, _load_resnet]:
        if loader():
            return
    print("[WARN] Face model 없음 — ArcFace Cosine은 NaN으로 출력됩니다.")


def _load_facenet() -> bool:
    global _face_model, _face_model_name
    try:
        from facenet_pytorch import InceptionResnetV1
        _face_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        _face_model_name = "facenet"
        print("FaceNet (VGGFace2) loaded.")
        return True
    except Exception:
        return False


def _load_resnet() -> bool:
    global _face_model, _face_model_name
    try:
        import torchvision.models as tvm
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        _face_model = m.eval().to(DEVICE)
        _face_model_name = "resnet50"
        print("ResNet50 (ImageNet) loaded as ArcFace fallback.")
        return True
    except Exception:
        return False


def _face_embed(img: np.ndarray):
    if _face_model is None or img.shape[0] < 32 or img.shape[1] < 32:
        return None
    size = 160 if _face_model_name == "facenet" else 224
    t = torch.from_numpy(cv2.resize(img, (size, size))).float().permute(2, 0, 1) / 255.0
    if _face_model_name == "facenet":
        t = (t - 0.5) / 0.5
    else:
        t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    with torch.no_grad():
        e = _face_model(t.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    return e / (np.linalg.norm(e) + 1e-8)


def arcface_cosine(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    if _face_model is None:
        return float("nan")
    face = matte < 127
    p_crop, fc = _bbox_crop(pred, face)
    g_crop, _  = _bbox_crop(gt,   face)
    ep, eg = _face_embed(p_crop), _face_embed(g_crop)
    if ep is None or eg is None:
        return float("nan")
    return float(np.dot(ep, eg))


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(real_imgs: list, fake_imgs: list, dims: int = 2048) -> float:
    """표준 InceptionV3 FID. dims=2048 (pool3 2048-dim 공분산).

    주의: 2048-dim 공분산은 표본 수 n < 2048 이면 rank-deficient → 통합(573)에서만
    안정적. braid 단독(n=107) 분리 보고 금지 (스펙 보고 단위 규칙).
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("[WARN] pytorch_fid 없음 — FID는 NaN. pip install pytorch-fid")
        return float("nan")
    with tempfile.TemporaryDirectory() as tmp:
        real_dir, fake_dir = Path(tmp) / "real", Path(tmp) / "fake"
        real_dir.mkdir(); fake_dir.mkdir()
        for i, img in enumerate(real_imgs):
            if img is not None:
                Image.fromarray(img).save(real_dir / f"{i:05d}.png")
        for i, img in enumerate(fake_imgs):
            if img is not None:
                Image.fromarray(img).save(fake_dir / f"{i:05d}.png")
        nr = len(list(real_dir.glob("*.png")))
        nf = len(list(fake_dir.glob("*.png")))
        if nr < 2 or nf < 2:
            return float("nan")
        return float(fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=32, device=str(DEVICE), dims=dims, num_workers=0,
        ))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# 최종 정량 지표 7개 (스펙 확정).
#   (label, per_image_key, fid_key, higher_better, axis, paired, unit)
#     unit = "split"    → per-image. braid/unbraid/macro 분리 보고 (braid는 ±CI)
#     unit = "combined" → FID. 통합(573)만 보고 (2048-dim 공분산, braid n=107 rank-deficient)
SPEC_METRICS = [
    ("Sketch LPIPS ↓",      "sketch_lpips", None,       False, "구조",       "paired(vs sketch)", "split"),
    ("Edge IoU ↑",          "edge_iou",     None,       True,  "구조(보조)", "paired(vs sketch)", "split"),
    ("Hair FID ↓",          None,           "hair_fid", False, "리얼리즘",   "unpaired",          "combined"),
    ("LPIPS (GT) ↓",        "lpips",        None,       False, "외형",       "paired(vs GT)",     "split"),
    ("Boundary FID ↓",      None,           "bnd_fid",  False, "경계",       "unpaired",          "combined"),
    ("Boundary LPIPS ↓",    "bnd_lpips",    None,       False, "경계(보조)", "paired(vs GT)",     "split"),
    ("Full-portrait FID ↓", None,           "full_fid", False, "전역 조화",  "unpaired",          "combined"),
]

PER_IMAGE_KEYS = ["sketch_lpips", "edge_iou", "lpips", "bnd_lpips"]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def _safe_mean(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vs) / len(vs) if vs else None


def _ci95(vals):
    """95% 신뢰구간 half-width (1.96·SD/√n). 표본<2면 None."""
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vs)
    if n < 2:
        return None
    m = sum(vs) / n
    var = sum((v - m) ** 2 for v in vs) / (n - 1)
    return 1.96 * math.sqrt(var) / math.sqrt(n)


def hair_masked_lpips(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    """hair 영역 bbox crop + 마스킹 후 LPIPS (vs GT)."""
    hair = matte > 127
    p_crop, mc = _bbox_crop(pred, hair)
    g_crop, _  = _bbox_crop(gt,   hair)
    p_crop[~mc] = 0
    g_crop[~mc] = 0
    if p_crop.shape[0] < 8 or p_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(p_crop, g_crop)


def build_summary(rows_braid, rows_unbraid, fid: dict) -> list[dict]:
    """SPEC_METRICS 각 지표를 보고단위별로 정리.
    combined 모드: rows_braid, rows_unbraid 둘 다 지정. single split: 한쪽만(나머지 None)."""
    out = []
    for label, pk, fk, higher, axis, paired, unit in SPEC_METRICS:
        r = {"label": label, "axis": axis, "paired": paired, "unit": unit,
             "braid": None, "braid_ci95": None, "unbraid": None, "unbraid_ci95": None,
             "macro": None, "macro_ci95": None, "combined": None}
        if unit == "combined":
            r["combined"] = fid.get(fk)
        else:
            if rows_braid is not None:
                r["braid"]      = _safe_mean([x.get(pk) for x in rows_braid])
                r["braid_ci95"] = _ci95([x.get(pk) for x in rows_braid])
            if rows_unbraid is not None:
                r["unbraid"]      = _safe_mean([x.get(pk) for x in rows_unbraid])
                r["unbraid_ci95"] = _ci95([x.get(pk) for x in rows_unbraid])
            if r["braid"] is not None and r["unbraid"] is not None:
                r["macro"] = (r["braid"] + r["unbraid"]) / 2
                # 독립 두 그룹 평균의 평균 → CI = ½·√(CI_b² + CI_u²)
                cb, cu = r["braid_ci95"], r["unbraid_ci95"]
                if cb is not None and cu is not None:
                    r["macro_ci95"] = 0.5 * math.sqrt(cb ** 2 + cu ** 2)
        out.append(r)
    return out


def print_summary(summary: list[dict], tag: str, n_b: int, n_u: int):
    print(f"\n{'='*80}")
    print(f"  tag={tag}   braid n={n_b}   unbraid n={n_u}   combined N={n_b + n_u}")
    print("=" * 80)
    def _mc(v, ci):
        s = _fmt(v)
        return f"{s}±{ci:.4f}" if (v is not None and ci is not None) else s

    print(f"  {'Metric':<18}{'축':<10}{'braid(±CI95)':>22}{'unbraid(±CI95)':>22}{'macro(±CI95)':>22}{'comb573':>10}")
    print("-" * 100)
    for r in summary:
        if r["unit"] == "combined":
            braid = unbraid = macro = "—"
            comb = _fmt(r["combined"])
        else:
            braid   = _mc(r["braid"],   r["braid_ci95"])
            unbraid = _mc(r["unbraid"], r["unbraid_ci95"])
            macro   = _mc(r["macro"],   r["macro_ci95"])
            comb = "—"
        print(f"  {r['label']:<18}{r['axis']:<10}{braid:>22}{unbraid:>22}{macro:>22}{comb:>10}")
    print("=" * 100)
    print("  · FID(③⑤⑦)=통합573만  · per-image(①②④⑥)=braid/unbraid/macro (모두 ±CI95)")
    print("  · run 간 유의차 확정은 scripts/stats_compare.py (paired t-test + Wilcoxon)")


def write_summary_csv(summary: list[dict], tag: str, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "axis", "paired", "unit",
                    "braid", "braid_ci95", "unbraid", "unbraid_ci95",
                    "macro", "macro_ci95", "combined573"])
        ci = lambda v: (f"{v:.4f}" if v is not None else "N/A")
        for r in summary:
            w.writerow([
                r["label"], r["axis"], r["paired"], r["unit"],
                _fmt(r["braid"]),   ci(r["braid_ci95"]),
                _fmt(r["unbraid"]), ci(r["unbraid_ci95"]),
                _fmt(r["macro"]),   ci(r["macro_ci95"]),
                _fmt(r["combined"]),
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_stems(pred_dir: Path, gt_dir: Path, suffix: str) -> list[str]:
    pred_stems = {
        (p.stem[:-len(suffix)] if suffix and p.stem.endswith(suffix) else p.stem)
        for p in pred_dir.glob("*.png")
    }
    gt_stems = {p.stem for p in gt_dir.glob("*.png")}
    common = pred_stems & gt_stems
    if not common:
        print("[ERROR] pred와 GT 사이에 공통 stem이 없습니다.")
        sys.exit(1)
    skipped = gt_stems - common
    if skipped:
        print(f"[INFO] GT에만 있어 제외된 stem: {len(skipped)}개")
    return sorted(common)


def _process_split(split_name: str, pred_dir: Path, suffix: str) -> dict:
    """단일 split 평가. per-image rows(split 라벨 포함) + FID용 이미지 리스트 dict 반환."""
    ds     = SPLITS[split_name]
    gt_dir = ds["img"]
    mt_dir = ds["matte"]
    sk_dir = ds["sketch"]

    stems = discover_stems(pred_dir, gt_dir, suffix)
    print(f"  [{split_name}] {len(stems)}개")

    rows = []
    hair_r, hair_f = [], []   # Hair FID  (hair 마스킹/크롭)
    bnd_r,  bnd_f  = [], []   # Boundary FID (경계 band)
    full_r, full_f = [], []   # Full-portrait FID (whole-image)

    for stem in tqdm(stems, desc=f"[{split_name}]"):
        pred_name = f"{stem}{suffix}.png" if suffix else f"{stem}.png"
        pred_path = pred_dir / pred_name

        gt    = np.array(Image.open(gt_dir / f"{stem}.png").convert("RGB"))
        matte = np.array(Image.open(mt_dir / f"{stem}.png").convert("L"))
        sk    = np.array(Image.open(sk_dir / f"{stem}.png").convert("RGB"))

        if not pred_path.exists():
            rows.append({"stem": stem, "split": split_name, **{k: None for k in PER_IMAGE_KEYS}})
            continue

        pred = np.array(Image.open(pred_path).convert("RGB"))
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        pred_e = canny_edges(pred)
        sk_e   = canny_edges(sk)
        hair   = matte > 127
        bnd    = get_boundary_mask(matte)

        rows.append({
            "stem":         stem,
            "split":        split_name,
            "sketch_lpips": sketch_lpips(pred, sk, matte),         # 구조 (vs sketch)
            "edge_iou":     edge_iou(pred_e, sk_e, matte),         # 구조 보조 (vs sketch)
            "lpips":        hair_masked_lpips(pred, gt, matte),    # 외형 (vs GT, hair-masked)
            "bnd_lpips":    boundary_lpips(pred, gt, matte),       # 경계 보조 (vs GT)
        })

        hair_r.append(extract_region_crop(gt,   hair))
        hair_f.append(extract_region_crop(pred, hair))
        bnd_r.append(extract_region_crop(gt,   bnd, min_px=16))
        bnd_f.append(extract_region_crop(pred, bnd, min_px=16))
        full_r.append(gt)
        full_f.append(pred)

    return {"rows": rows, "hair_r": hair_r, "hair_f": hair_f,
            "bnd_r": bnd_r, "bnd_f": bnd_f, "full_r": full_r, "full_f": full_f}


def _compute_fids(parts: dict) -> dict:
    """통합(573) FID 3종 계산 (dims=2048)."""
    print("\nFID 계산 중 (통합, dims=2048)...")
    fid = {
        "hair_fid": compute_fid([x for x in parts["hair_r"] if x is not None],
                                [x for x in parts["hair_f"] if x is not None]),
        "bnd_fid":  compute_fid([x for x in parts["bnd_r"]  if x is not None],
                                [x for x in parts["bnd_f"]  if x is not None]),
        "full_fid": compute_fid([x for x in parts["full_r"] if x is not None],
                                [x for x in parts["full_f"] if x is not None]),
    }
    for k in ("hair_fid", "bnd_fid", "full_fid"):
        print(f"  {k}: {_fmt(fid[k])}")
    return fid


def _merge_parts(pb: dict, pu: dict) -> dict:
    return {k: pb[k] + pu[k] for k in ("hair_r", "hair_f", "bnd_r", "bnd_f", "full_r", "full_f")}


def _n_valid(rows):
    return sum(1 for r in rows if r.get("sketch_lpips") is not None)


def _write_per_image(rows, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "split"] + PER_IMAGE_KEYS)
        w.writeheader()
        w.writerows(rows)


def _save_and_report(summary, rows, tag, out, n_b, n_u):
    print_summary(summary, tag, n_b, n_u)
    per_path = Path(f"{out}_per_image.csv"); _write_per_image(rows, per_path)
    sum_path = Path(f"{out}_summary.csv");  write_summary_csv(summary, tag, sum_path)
    print(f"\n저장 완료:\n  {per_path}\n  {sum_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate braid/unbraid/combined split (7-metric spec).")
    parser.add_argument("--split",       required=True, choices=list(SPLITS.keys()),
                        help="'braid', 'unbraid', 'combined'")
    parser.add_argument("--pred",        required=True, nargs="+",
                        help="생성 이미지 디렉토리. combined는 'braid_dir unbraid_dir' 순서로 2개 지정")
    parser.add_argument("--pred-suffix", default="",
                        help="pred 파일명 접미사 (예: _full  →  stem_full.png)")
    parser.add_argument("--out",         default=None,
                        help="출력 경로 prefix (기본: eval_results/{split}_{pred_dir_name})")
    parser.add_argument("--tag",         default=None,
                        help="결과 테이블 컬럼 레이블")
    args = parser.parse_args()

    suffix = args.pred_suffix

    if args.split == "combined":
        if len(args.pred) == 1:
            braid_dir = unbraid_dir = Path(args.pred[0])
            if not braid_dir.exists():
                print(f"[ERROR] pred 디렉토리 없음: {braid_dir}"); sys.exit(1)
        elif len(args.pred) == 2:
            braid_dir, unbraid_dir = Path(args.pred[0]), Path(args.pred[1])
            for p in (braid_dir, unbraid_dir):
                if not p.exists():
                    print(f"[ERROR] pred 디렉토리 없음: {p}"); sys.exit(1)
        else:
            print("[ERROR] --split combined은 --pred를 1개(단일 폴더) 또는 2개(braid unbraid) 지정하세요")
            sys.exit(1)

        tag = args.tag or braid_dir.name
        out = Path(args.out) if args.out else ROOT / "eval_results" / f"combined_{braid_dir.name}"
        out.parent.mkdir(parents=True, exist_ok=True)

        print("split  : combined (braid + unbraid)")
        print(f"pred   : {braid_dir}" + (f"  |  {unbraid_dir}" if braid_dir != unbraid_dir else " (단일 폴더, 자동 분리)"))
        print(f"suffix : '{suffix}'")
        print(f"out    : {out}_*\n")

        print("braid 처리 중...")
        pb = _process_split("braid",   braid_dir,   suffix)
        print("unbraid 처리 중...")
        pu = _process_split("unbraid", unbraid_dir, suffix)

        fid = _compute_fids(_merge_parts(pb, pu))   # FID는 통합(573)만
        summary = build_summary(pb["rows"], pu["rows"], fid)
        _save_and_report(summary, pb["rows"] + pu["rows"], tag,
                         out, _n_valid(pb["rows"]), _n_valid(pu["rows"]))

    else:
        if len(args.pred) != 1:
            print("[ERROR] braid/unbraid split은 --pred를 1개만 지정하세요"); sys.exit(1)
        pred_dir = Path(args.pred[0])
        if not pred_dir.exists():
            print(f"[ERROR] pred 디렉토리 없음: {pred_dir}"); sys.exit(1)
        if args.split == "braid":
            print("[WARN] braid 단독 FID는 2048-dim에서 rank-deficient(n=107<2048). "
                  "FID는 통합(573, --split combined)으로 보고 권장.")

        tag = args.tag or pred_dir.name
        out = Path(args.out) if args.out else ROOT / "eval_results" / f"{args.split}_{pred_dir.name}"
        out.parent.mkdir(parents=True, exist_ok=True)

        ds = SPLITS[args.split]
        print(f"split  : {args.split}")
        print(f"pred   : {pred_dir}")
        print(f"suffix : '{suffix}'")
        print(f"gt     : {ds['img']}")
        print(f"out    : {out}_*\n")

        parts = _process_split(args.split, pred_dir, suffix)
        fid = _compute_fids(parts)
        rb, ru = (parts["rows"], None) if args.split == "braid" else (None, parts["rows"])
        summary = build_summary(rb, ru, fid)
        n = _n_valid(parts["rows"])
        _save_and_report(summary, parts["rows"], tag, out,
                         n if args.split == "braid" else 0,
                         n if args.split == "unbraid" else 0)


if __name__ == "__main__":
    main()
