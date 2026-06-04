# Hair-DiT Forward: 모델 아키텍처 & 실험 설계

## 1. 모델 아키텍처

SD3.5-medium 기반 ControlNet으로 sketch + matte → hair region을 생성한다.

### 입력 conditioning (ctrl_cond, 17ch)

```
sketch (B,3,512,512) → frozen VAE encode → sketch_latent (B,16,64,64)
matte  (B,1,512,512) → bilinear downsample → matte_latent (B,1,64,64)
ctrl_cond = cat([sketch_latent, matte_latent], dim=1)   # (B, 17, 64, 64)
```

- **MatteCNN 없음.** matte는 bilinear interpolation으로만 다운샘플.
- ch 1–16: sketch VAE latent (구조 정보)
- ch 17: raw matte downsample (명시적 공간 마스크)
- `pos_embed_input`은 17ch zero-conv 초기화 (표준 ControlNet).

### ControlNet 블록 (12개, depth-aligned warm-start)

- ControlNet 블록 i가 생성한 residual은 transformer 블록 **2i, 2i+1**에 주입된다
  (diffusers `interval_control = 24/12 = 2`). 12개 residual이 24블록 전체를 커버.
- 블록 i는 상대 depth `i/12 == 2i/24`로 주입 지점과 같은 깊이에 위치 →
  **transformer 블록 2i에서 가중치를 warm-start**하여 init 깊이를 주입 깊이에 정렬.
- SD3.5는 앞쪽 블록에만 dual attention이 있어, 깊은 even 소스(14~22)는 attn2/큰
  norm1이 없다. 이 부분은 `from_transformer`의 sequential 복사값(블록 i)을 유지 →
  **fresh init 없이** 모든 파라미터가 실제 학습된 transformer 가중치.

### Forward

```
1. ctrl_cond 구성 (위)
2. SD3ControlNetModel(noisy_latent, ctrl_cond, null_emb, sigma) → 12 block residuals
3. (optional) matte token gate: residual을 matte 영역만 남기고 곱셈
4. frozen SD3.5 transformer(noisy_latent, residuals) → v_pred
5. flow matching loss + LPIPS + edge
```

- text encoder 미사용 → learned null embedding (`null_encoder_hidden_states`,
  `null_pooled_projections`)을 nn.Parameter로 학습.
- VAE, transformer는 frozen. ControlNet + null embedding만 trainable.

### Inference (BLD)

face 이미지가 주어지면 Blended Latent Diffusion으로 full image를 직접 생성:
매 denoising 스텝마다 matte 바깥 영역을 원본 face의 noised latent로 덮어써
배경을 디퓨전 trajectory 안에서 보존한다.

---

## 2. 실험 설계 (4가지)

Matte는 두 경로로 작동할 수 있다:

- **Matte conditioning**: ctrl_cond의 17번째 채널(raw matte)
- **Matte gate**: residual을 transformer에 주입할 때 token-level matte mask로 게이팅

|                    | Matte gate ✗ | Matte gate ✓ |
|--------------------|:------------:|:------------:|
| **Matte cond. ✓**  |   **exp3**   |   **exp1**   |
| **Matte cond. ✗**  |   **exp4**   |   **exp2**   |

### 조건별 정의

| 실험 | ctrl_cond ch17 | matte_gate | config flag |
|------|----------------|:----------:|-------------|
| **exp1** | 실제 matte | ✓ | `matte_gate: true` |
| **exp2** | zeros | ✓ | `zero_matte_cond: true`, `matte_gate: true` |
| **exp3** | 실제 matte | ✗ | (기준선) |
| **exp4** | zeros | ✗ | `zero_matte_cond: true` |

- `zero_matte_cond: true` → ctrl_cond 17번째 채널을 `torch.zeros`로 대체 (ch 1–16 동일).
- `matte_gate: true` → block residual을 matte token mask(area pool, 32×32)로 곱해
  hair 영역 토큰만 제어 신호 주입 (바깥 = 0). 학습·추론 양쪽 적용.

### 비교 축

- **exp3 vs exp1** → matte gate 효과 (matte cond 고정)
- **exp3 vs exp4** → matte conditioning 효과 (gate 없음 고정)
- **exp1 vs exp2** → matte conditioning 효과 (gate 있음 고정)
- **exp2 vs exp4** → matte gate 효과 (matte cond 없음 고정)

### 예상 민감 지표

1. **Boundary LPIPS** — matte gate 효과가 가장 잘 드러남
2. **Face LPIPS / ArcFace** — matte conditioning 없을 때 배경 침범 여부
3. **Hair LPIPS(GT), SSIM** — 전반적 생성 품질

---

## 3. 학습 절차

각 실험은 Phase 1 (unbraid pretrain) → Phase 2 (braid finetune) 2단계.

### Phase 1
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/exp1_phase1.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/exp2_phase1.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/exp3_phase1.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/exp4_phase1.yaml
```

### Phase 2 (Phase 1 final.pth에서 resume)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/exp1_phase2.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/exp2_phase2.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/exp3_phase2.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/exp4_phase2.yaml
```

- batch_size 16, grad_accum 1, save_every 10 epoch
- LPIPS는 전체 스텝의 30% 지점부터 활성화 (이때 loss_total 점프는 정상)

### config 상속

```
exp{N}_phase1.yaml → phase1_unbraid.yaml → base.yaml
exp{N}_phase2.yaml → phase2_braid.yaml  → base.yaml
```

### 0604
다음 실험
ctrl_cond 구성

- sketch_latent : 16채널
- raw_matte : 1채널

ControlNet 구성
- front12개
