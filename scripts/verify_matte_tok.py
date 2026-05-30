"""
matte_tok 토큰 순서 검증 스크립트.

make_matte_tok()으로 만든 (B,1024,1)을 32x32로 reshape했을 때
원본 matte를 32x32로 bilinear downsample한 것과 일치하는지 확인한다.

Usage:
  python scripts/verify_matte_tok.py
  python scripts/verify_matte_tok.py --matte path/to/matte.png
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.controlnet_sd35 import make_matte_tok

DEFAULT_MATTE = "dataset/unbraid/matte/train/Fgo_350.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matte", default=DEFAULT_MATTE)
    parser.add_argument("--output", default="outputs/verify_matte_tok.png")
    args = parser.parse_args()

    matte_path = Path(args.matte)
    assert matte_path.exists(), f"matte 파일 없음: {matte_path}"

    # 1. 원본 matte 로드 (1, 1, 512, 512)
    matte_pil = Image.open(matte_path).convert("L").resize((512, 512), Image.BILINEAR)
    matte = transforms.ToTensor()(matte_pil).unsqueeze(0)  # (1, 1, 512, 512)

    # 2. make_matte_tok 적용 → (1, 1024, 1)
    tok = make_matte_tok(matte)  # (1, 1024, 1)

    # 3. 32x32로 reshape
    tok_grid = tok.squeeze(-1).reshape(32, 32)  # (32, 32)

    # 4. 직접 bilinear downsample한 기준값
    ref_grid = F.interpolate(matte, size=(32, 32), mode="bilinear", align_corners=False)
    ref_grid = ref_grid.squeeze()  # (32, 32)

    # 5. 수치 비교
    diff = (tok_grid - ref_grid).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max  |tok - ref| = {max_diff:.2e}")
    print(f"mean |tok - ref| = {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("PASS: token order correct")
    else:
        print("FAIL: token order mismatch — fix make_matte_tok")

    # 6. 시각화 저장 (원본 32x32 | tok reshape | 차이 x10)
    def to_pil_32(t: torch.Tensor) -> Image.Image:
        arr = t.float().clamp(0, 1).numpy()
        return Image.fromarray((arr * 255).astype("uint8")).resize((256, 256), Image.NEAREST)

    ref_img  = to_pil_32(ref_grid)
    tok_img  = to_pil_32(tok_grid)
    diff_img = to_pil_32((diff * 10).clamp(0, 1))

    # 원본 512px도 256으로 리사이즈
    orig_img = matte_pil.resize((256, 256), Image.BILINEAR)

    combined = Image.new("L", (256 * 4, 256))
    combined.paste(orig_img,  (0,   0))
    combined.paste(ref_img,   (256, 0))
    combined.paste(tok_img,   (512, 0))
    combined.paste(diff_img,  (768, 0))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.save(args.output)
    print(f"\n저장: {args.output}")
    print("컬럼: [원본 512→256] [ref 32x32] [tok reshape] [차이 x10]")
    print("      세 컬럼이 시각적으로 동일하면 토큰 순서 정상")


if __name__ == "__main__":
    main()
