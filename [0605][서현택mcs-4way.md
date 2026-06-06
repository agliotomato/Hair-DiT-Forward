# MCS 4-way Ablation 실험 설계

브랜치: `feature/mcs-4way`

---

## ControlNet 구성

### ctrl_cond (17ch)

```
matte (B,1,512,512)
  ├─ MatteCNN (trainable)  →  matte_feat  (B,16,64,64)
  └─ area-avg downsample   →  matte_latent (B,1,64,64)

sketch (B,3,512,512)
  └─ frozen VAE encode     →  sketch_latent (B,16,64,64)

ctrl_cond = cat( [sketch_latent ⊕ matte_feat,  matte_latent],  dim=1 )
                  └────────── 16ch ──────────   ───── 1ch ────  = 17ch
```

- `sketch_latent ⊕ matte_feat`: element-wise addition
- `matte_latent`: 512→64 area-average pooling (구역별 평균, bilinear 아님)

### MatteCNN

```
입력  1ch  512×512
  Conv(k=3, s=2) → GroupNorm → SiLU  16ch  256×256
  Conv(k=3, s=2) → GroupNorm → SiLU  32ch  128×128
  Conv(k=3, s=2) → GroupNorm → SiLU  16ch   64×64
출력 16ch   64×64
```

### Gate (optional)

block_samples의 image token에 soft matte 마스크 곱셈:

```
matte_tok = area-avg(matte → 32×32) → flatten → (B, 1024, 1)
block_samples[k][:, :1024, :] *= matte_tok
```

`schedule` config으로 적용 범위 제어: `none` | `all` | `front_only` | `back_only`

---

## 4-way Ablation

|                       | gate ✗ (`schedule: none`) | gate ✓ (`schedule: all`) |
|-----------------------|:-------------------------:|:------------------------:|
| **matte cond ✓**      | **mcs1**                  | **mcs2**                 |
| **matte cond ✗**      | **mcs3**                  | **mcs4**                 |

- **matte cond ✓** (`zero_matte_cond: false`): 실제 matte 사용 — `matte_feat` + `matte_latent` 모두 real
- **matte cond ✗** (`zero_matte_cond: true`): `matte_feat`, `matte_latent` 모두 zeros → ctrl_cond = sketch 전용

### 비교 축

| 비교 | 고정 | 측정 효과 |
|------|------|-----------|
| mcs1 vs mcs2 | matte cond ✓ | gate 효과 |
| mcs3 vs mcs4 | matte cond ✗ | gate 효과 (cond 없을 때) |
| mcs1 vs mcs3 | gate ✗ | matte conditioning 효과 |
| mcs2 vs mcs4 | gate ✓ | matte conditioning 효과 (gate 있을 때) |

---

## Config 파일

| 실험 | Phase 1 | Phase 2 |
|------|---------|---------|
| mcs1 | `configs/mcs1_phase1.yaml` | `configs/mcs1_phase2.yaml` |
| mcs2 | `configs/mcs2_phase1.yaml` | `configs/mcs2_phase2.yaml` |
| mcs3 | `configs/mcs3_phase1.yaml` | `configs/mcs3_phase2.yaml` |
| mcs4 | `configs/mcs4_phase1.yaml` | `configs/mcs4_phase2.yaml` |

Config 상속: `mcs{N}_phase{1,2}.yaml` → `phase{1,2}_{unbraid,braid}.yaml` → `base.yaml`

---

## 학습 명령어

```bash
# Phase 1
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --config configs/mcs1_phase1.yaml
CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py --config configs/mcs2_phase1.yaml
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --config configs/mcs3_phase1.yaml
CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py --config configs/mcs4_phase1.yaml

# Phase 2 (Phase 1 epoch_40 완료 후)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --config configs/mcs1_phase2.yaml
CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py --config configs/mcs2_phase2.yaml
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --config configs/mcs3_phase2.yaml
CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py --config configs/mcs4_phase2.yaml
```

## 추론 명령어

```
CUDA_VISIBLE_DEVICES=0 python3 scripts/infer_custom.py \
  --sketch     dataset/braid/sketch/test/ \
  --matte      dataset/braid/matte/test/ \
  --face       dataset/braid/img/test/ \
  --checkpoint checkpoints/mcs1_phase2/final.pth \
  --config     configs/mcs1_phase2.yaml \
  --output_dir results/mcs1
```
---

## 평가 명령어
```
python3 scripts/eval_metrics.py \
  --split combined \
  --pred  results/mcs1_braid results/mcs1_unbraid \
  --tag   mcs1
```


## 평가 결과

| Metric | mcs1 | mcs2 | mcs3 | mcs4 |
|--------|------|------|------|------|
| Edge IoU ↑ | 0.0640 | 0.0643 | 0.0628 | 0.0628 |
| Chamfer Dist ↓ | 4.7013 | 4.6655 | 4.8089 | 4.8726 |
| Sketch LPIPS ↓ | 0.7599 | 0.7679 | 0.7567 | 0.7416 |
| Hair FID ↓ | 1.1078 | 1.5370 | 1.5352 | 1.9295 |
| LPIPS (GT) ↓ | 0.2973 | 0.3314 | 0.3066 | 0.3282 |
| SSIM (GT) ↑ | 0.5893 | 0.5891 | 0.5866 | 0.5906 |
| PSNR (GT) ↑ | 11.6486 | 11.4002 | 11.5611 | 11.4483 |
| Boundary FID ↓ | 0.0121 | 0.0174 | 0.0108 | 0.0204 |
| Boundary LPIPS ↓ | 0.0197 | 0.0214 | 0.0220 | 0.0227 |
| Face LPIPS ↓ | 0.0033 | 0.0037 | 0.0033 | 0.0037 |
| ArcFace Cos ↑ | 0.6860 | 0.6733 | 0.7014 | 0.6685 |
