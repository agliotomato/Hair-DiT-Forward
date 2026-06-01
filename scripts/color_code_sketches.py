"""
Color-code raw (rainbow-label) sketches to real hair colors — inference 전처리.

학습 때 StrokeColorSampler(p=1.0)가 매 iteration 적용했던 것과 동일한 변환을
추론 입력에 deterministic(평균색)하게 한 번 적용한다. raw 스케치는 stroke마다
임의 고유색(라벨)이라, 그대로 모델에 넣으면 학습 분포와 어긋난다. 각 stroke 라벨을
img*matte의 실제 머리색 평균으로 교체해서 학습이 본 color-coded 스케치를 재현한다.

Usage:
  python scripts/color_code_sketches.py \
    --sketch dataset/braid/sketch/test \
    --img    dataset/braid/img/test \
    --matte  dataset/braid/matte/test \
    --output dataset/braid/sketch_colorcoded/test

  # 이후 추론은 이 폴더를 --sketch로:
  python scripts/infer_custom.py --sketch dataset/braid/sketch_colorcoded/test \
    --matte dataset/braid/matte/test --face dataset/braid/img/test ...
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def color_code(
    sketch: np.ndarray,   # (H, W, 3) uint8
    img: np.ndarray,      # (H, W, 3) uint8
    matte: np.ndarray,    # (H, W)    uint8 [0,255]
    quantize_bits: int = 5,
    min_pixels: int = 10,
) -> np.ndarray:
    """각 stroke 라벨 → img(매트 내) 실제 머리색 평균으로 교체. (augmentation.StrokeColorSampler 재현)"""
    shift = 8 - quantize_bits
    sk_q = (sketch >> shift) << shift                  # (H, W, 3) 양자화 라벨
    m = (matte.astype(np.float32) / 255.0)[..., None]  # (H, W, 1)
    target = img.astype(np.float32) * m                # hair 영역 (배경 0)

    out = sketch.copy()
    flat = sk_q.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)

    for color in uniq:
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        if r == 0 and g == 0 and b == 0:               # 검은 배경 stroke는 유지
            continue
        mask = (sk_q[..., 0] == r) & (sk_q[..., 1] == g) & (sk_q[..., 2] == b)  # (H, W)
        hair = target[mask]                            # (N, 3)
        valid = hair.sum(axis=1) > (0.05 * 255)        # 매트 내 유효 픽셀
        if valid.sum() < min_pixels:
            continue
        col = hair[valid].mean(axis=0)                 # (3,) 평균 머리색
        out[mask] = col.astype(np.uint8)
    return out


def _match(d: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg"):
        c = d / (stem + ext)
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", required=True, help="raw (rainbow-label) 스케치 디렉토리")
    ap.add_argument("--img",    required=True, help="원본 사진 디렉토리")
    ap.add_argument("--matte",  required=True, help="matte 디렉토리")
    ap.add_argument("--output", required=True, help="color-coded 스케치 저장 디렉토리")
    ap.add_argument("--quantize_bits", type=int, default=5)
    ap.add_argument("--min_pixels",    type=int, default=10)
    args = ap.parse_args()

    sk_dir, img_dir, mt_dir = Path(args.sketch), Path(args.img), Path(args.matte)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sk_dir.glob("*.png")) + sorted(sk_dir.glob("*.jpg"))
    print(f"{len(files)}장 처리 → {out_dir}")

    done = 0
    for sf in files:
        stem = sf.stem
        imgf = _match(img_dir, stem)
        mtf = _match(mt_dir, stem)
        if imgf is None or mtf is None:
            print(f"  [skip] {stem}: img/matte 없음")
            continue
        sketch = np.array(Image.open(sf).convert("RGB"))
        img = np.array(Image.open(imgf).convert("RGB").resize(sketch.shape[1::-1]))
        matte = np.array(Image.open(mtf).convert("L").resize(sketch.shape[1::-1]))
        coded = color_code(sketch, img, matte, args.quantize_bits, args.min_pixels)
        Image.fromarray(coded).save(out_dir / f"{stem}.png")
        done += 1

    print(f"완료: {done}장 저장 → {out_dir}")


if __name__ == "__main__":
    main()
