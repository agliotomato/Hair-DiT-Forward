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

Metrics:
  [1] Sketch Fidelity   : Edge IoU ↑,  Chamfer Distance ↓,  Sketch LPIPS ↓
  [2] Generation Quality: Hair FID ↓,  LPIPS(GT) ↓,  SSIM(GT) ↑,  PSNR(GT) ↑
  [3] Boundary Quality  : Boundary FID ↓,  Boundary LPIPS ↓
  [4] Identity          : Face LPIPS ↓,  ArcFace Cosine ↑

Outputs:
  <out>_per_image.csv
  <out>_summary.csv
  console table + LaTeX tabular
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

def compute_fid(real_imgs: list, fake_imgs: list) -> float:
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
            batch_size=32, device=str(DEVICE), dims=2048, num_workers=0,
        ))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

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

PER_IMAGE_KEYS = [
    "edge_iou", "chamfer", "sketch_lpips",
    "psnr", "ssim", "lpips",
    "bnd_lpips", "face_lpips", "arcface_cos",
]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def _safe_mean(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vs) / len(vs) if vs else None


def print_table(rows: list, fid: dict, split: str, tag: str, n: int):
    W = 14
    print(f"\n{'='*52}")
    print(f"  [{split.upper()}]  tag={tag}  n={n}")
    print("=" * 52)
    print(f"  {'Metric':<24}  {tag:>{W}}")
    print("=" * 52)
    for group, metrics in METRIC_GROUPS:
        print(f"\n--- {group} ---")
        for label, pk, fk, _ in metrics:
            v = fid.get(fk) if fk else _safe_mean([r.get(pk) for r in rows])
            print(f"  {label:<24}  {_fmt(v):>{W}}")
    print("=" * 52)


def print_latex(rows: list, fid: dict, tag: str):
    print(f"\n% LaTeX — {tag}")
    print("\\begin{tabular}{lc}")
    print("\\hline")
    print(f"Metric & {tag} \\\\")
    print("\\hline")
    for group, metrics in METRIC_GROUPS:
        print(f"\\multicolumn{{2}}{{l}}{{\\textit{{{group}}}}} \\\\")
        for label, pk, fk, _ in metrics:
            v = fid.get(fk) if fk else _safe_mean([r.get(pk) for r in rows])
            clean = label.replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$")
            print(f"  {clean} & {_fmt(v)} \\\\")
    print("\\hline")
    print("\\end{tabular}")


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


def _process_split(
    split_name: str,
    pred_dir: Path,
    suffix: str,
) -> tuple[list[dict], list, list, list, list]:
    """단일 split 평가. (rows, fid_hr, fid_hf, fid_br, fid_bf) 반환."""
    ds     = SPLITS[split_name]
    gt_dir = ds["img"]
    mt_dir = ds["matte"]
    sk_dir = ds["sketch"]

    stems = discover_stems(pred_dir, gt_dir, suffix)
    print(f"  [{split_name}] {len(stems)}개")

    rows = []
    fid_hr, fid_hf = [], []
    fid_br, fid_bf = [], []

    for stem in tqdm(stems, desc=f"[{split_name}]"):
        pred_name = f"{stem}{suffix}.png" if suffix else f"{stem}.png"
        pred_path = pred_dir / pred_name

        gt    = np.array(Image.open(gt_dir / f"{stem}.png").convert("RGB"))
        matte = np.array(Image.open(mt_dir / f"{stem}.png").convert("L"))
        sk    = np.array(Image.open(sk_dir / f"{stem}.png").convert("RGB"))

        if not pred_path.exists():
            rows.append({"stem": stem, **{k: None for k in PER_IMAGE_KEYS}})
            continue

        pred = np.array(Image.open(pred_path).convert("RGB"))
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        pred_e = canny_edges(pred)
        sk_e   = canny_edges(sk)
        hair   = matte > 127
        bnd    = get_boundary_mask(matte)
        gq     = gen_quality_metrics(pred, gt, matte)

        rows.append({
            "stem":         stem,
            "edge_iou":     edge_iou(pred_e, sk_e, matte),
            "chamfer":      chamfer_distance(pred_e, sk_e, matte),
            "sketch_lpips": sketch_lpips(pred, sk, matte),
            "psnr":         gq["psnr"],
            "ssim":         gq["ssim"],
            "lpips":        gq["lpips"],
            "bnd_lpips":    boundary_lpips(pred, gt, matte),
            "face_lpips":   face_lpips(pred, gt, matte),
            "arcface_cos":  arcface_cosine(pred, gt, matte),
        })

        fid_hr.append(extract_region_crop(gt,   hair))
        fid_hf.append(extract_region_crop(pred, hair))
        fid_br.append(extract_region_crop(gt,   bnd, min_px=16))
        fid_bf.append(extract_region_crop(pred, bnd, min_px=16))

    return rows, fid_hr, fid_hf, fid_br, fid_bf


def _finalize(rows, fid_hr, fid_hf, fid_br, fid_bf, split_name, tag, out):
    print("\nFID 계산 중...")
    fid = {
        "hair_fid": compute_fid([x for x in fid_hr if x is not None],
                                [x for x in fid_hf if x is not None]),
        "bnd_fid":  compute_fid([x for x in fid_br if x is not None],
                                [x for x in fid_bf if x is not None]),
    }
    print(f"  Hair FID: {_fmt(fid['hair_fid'])}")
    print(f"  Bnd  FID: {_fmt(fid['bnd_fid'])}")

    valid = [r for r in rows if r.get("psnr") is not None]
    print_table(valid, fid, split_name, tag, len(valid))
    print_latex(valid, fid, tag)

    per_path = Path(f"{out}_per_image.csv")
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem"] + PER_IMAGE_KEYS)
        w.writeheader()
        w.writerows(rows)

    sum_path = Path(f"{out}_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", tag])
        for _, metrics in METRIC_GROUPS:
            for label, pk, fk, _ in metrics:
                v = fid.get(fk) if fk else _safe_mean([r.get(pk) for r in valid])
                w.writerow([label, _fmt(v)])

    print(f"\n저장 완료:\n  {per_path}\n  {sum_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate braid/unbraid/combined split.")
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
    try_load_face_model()

    if args.split == "combined":
        if len(args.pred) == 1:
            # 단일 폴더: braid/unbraid GT stem과 교차하여 자동 분리
            single_dir = Path(args.pred[0])
            if not single_dir.exists():
                print(f"[ERROR] pred 디렉토리 없음: {single_dir}")
                sys.exit(1)
            braid_dir   = single_dir
            unbraid_dir = single_dir
        elif len(args.pred) == 2:
            braid_dir   = Path(args.pred[0])
            unbraid_dir = Path(args.pred[1])
            for p in [braid_dir, unbraid_dir]:
                if not p.exists():
                    print(f"[ERROR] pred 디렉토리 없음: {p}")
                    sys.exit(1)
        else:
            print("[ERROR] --split combined은 --pred를 1개(단일 폴더) 또는 2개(braid unbraid) 지정하세요")
            sys.exit(1)

        tag = args.tag or braid_dir.name
        out = Path(args.out) if args.out else ROOT / "eval_results" / f"combined_{braid_dir.name}"
        out.parent.mkdir(parents=True, exist_ok=True)

        print(f"split  : combined (braid + unbraid)")
        print(f"pred   : {braid_dir}" + (f"  |  {unbraid_dir}" if braid_dir != unbraid_dir else " (단일 폴더, 자동 분리)"))
        print(f"suffix : '{suffix}'")
        print(f"out    : {out}_*\n")

        print("braid 처리 중...")
        r_b, hr_b, hf_b, br_b, bf_b = _process_split("braid",   braid_dir,   suffix)
        print("unbraid 처리 중...")
        r_u, hr_u, hf_u, br_u, bf_u = _process_split("unbraid", unbraid_dir, suffix)

        rows   = r_b  + r_u
        fid_hr = hr_b + hr_u
        fid_hf = hf_b + hf_u
        fid_br = br_b + br_u
        fid_bf = bf_b + bf_u

        _finalize(rows, fid_hr, fid_hf, fid_br, fid_bf, "combined", tag, out)

    else:
        if len(args.pred) != 1:
            print("[ERROR] braid/unbraid split은 --pred를 1개만 지정하세요")
            sys.exit(1)
        pred_dir = Path(args.pred[0])
        if not pred_dir.exists():
            print(f"[ERROR] pred 디렉토리 없음: {pred_dir}")
            sys.exit(1)

        tag = args.tag or pred_dir.name
        out = Path(args.out) if args.out else ROOT / "eval_results" / f"{args.split}_{pred_dir.name}"
        out.parent.mkdir(parents=True, exist_ok=True)

        ds = SPLITS[args.split]
        print(f"split  : {args.split}")
        print(f"pred   : {pred_dir}")
        print(f"suffix : '{suffix}'")
        print(f"gt     : {ds['img']}")
        print(f"out    : {out}_*\n")

        rows, fid_hr, fid_hf, fid_br, fid_bf = _process_split(args.split, pred_dir, suffix)
        _finalize(rows, fid_hr, fid_hf, fid_br, fid_bf, args.split, tag, out)


if __name__ == "__main__":
    main()
