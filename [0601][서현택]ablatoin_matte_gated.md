# Ablation: Matte-Gated Residual Schedule

## 1. 실험 설계

### 핵심 아이디어

SD3 ControlNet은 12개의 residual block을 생성하며, 이를 SD3 Transformer의 24개 블록에 stride-2로 주입한다.
기존 방식(`none`)은 모든 블록에 full residual을 주입하여 비-헤어 영역(얼굴, 배경)으로 정보가 번지는 **boundary bleeding** 문제가 발생할 수 있다.

이를 해결하기 위해 ControlNet residual에 **matte soft gate** (`residual × matte_tok`)를 적용, 주입 위치(block index)에 따라 schedule을 달리하는 ablation을 설계하였다.

> 입력 conditioning(MatteCNN feature + raw matte latent)은 모든 variant에서 **동일하게 유지**하고, gate schedule 하나만 변경하여 schedule 효과를 분리한다.

---

### 학습 구성

| 단계 | 데이터 | 공통 여부 |
|------|--------|----------|
| Phase 1 (Unbraid Pretraining) | unbraid | **모든 variant 공통** (schedule=none) |
| Phase 2 (Braid Finetuning) | braid | **variant별 별도 학습** |

Phase 2에서만 gate schedule을 적용하며, 동일한 Phase 1 체크포인트에서 출발하여 schedule 외 모든 조건을 통일하였다.

---

### Variant 정의

| Variant | Gate 적용 블록 | 설명 |
|---------|--------------|------|
| **none (베이스라인)** | 없음 | 기존 방식. Full residual을 모든 블록에 주입 |
| **front_only** | 0–11 | 전역 구조 단계에만 gate 적용 |
| **all** | 0–23 | 전체 블록에 gate 적용 |
| **back_only (Ours)** | 12–23 | 후반 정제 단계에만 gate 적용 |

**back_only 가설**: 전반부(0–11)는 전역 맥락(헤어 위치, 레이아웃)을 자유롭게 참조하고, 후반부(12–23)는 헤어 영역으로만 residual을 제한하여 boundary bleeding을 억제하면서 전역 구조를 보존한다.

---

### Matte Gate 구현

```python
# residual: (B, 1024+text, 1152)
# matte_tok: (B, 1024, 1) — matte를 32×32로 bilinear downsample 후 flatten
img = residual[:, :1024, :] * matte_tok   # 헤어 영역 비율만큼 scale
txt = residual[:, 1024:, :]               # text token 보존
gated = torch.cat([img, txt], dim=1)
```

gate는 hard 0/1이 아닌 **soft matte(0~1)** 값을 그대로 사용하여 경계 아티팩트를 방지한다.
학습과 추론 양쪽에 동일하게 적용한다.

---

## 2. 학습 Loss (40 epoch, final.pth 기준 추론)

| Variant | Train Loss (ep40) | Val Loss (ep40) | Best Val Loss |
|---------|:-----------------:|:---------------:|:-------------:|
| none | 0.0386 | 0.0808 | - |
| back_only | 0.0348 | 0.0832 | **0.0614** (ep5) |
| all | 0.0461 | 0.0752 | 0.0655 (ep5) |
| front_only | **0.0339** | 0.0861 | 0.0655 (ep5) |

> back_only가 best val loss 0.0614으로 가장 낮음. all은 train loss가 가장 높고 val loss는 중간.  
> none ep40 val loss(0.0808)는 back_only(0.0832), front_only(0.0861)보다 낮고 all(0.0752)보다 높음.

---

## 3. 정량 평가 결과 (braid test set)

굵은 값 = 해당 지표 최고

| Metric | none (베이스라인) | back_only (Ours) | all | front_only |
|--------|:---:|:---:|:---:|:---:|
| Edge IoU ↑ | 0.0999 | **0.1012** | 0.0994 | 0.1010 |
| Chamfer Dist ↓ | 2.8961 | 2.8845 | 2.9618 | **2.8372** |
| Sketch LPIPS ↓ | 0.7017 | 0.7033 | **0.6923** | 0.7092 |
| LPIPS (GT) ↓ | **0.1793** | 0.1831 | 0.1850 | 0.1797 |
| SSIM (GT) ↑ | 0.6094 | 0.6110 | 0.6056 | **0.6119** |
| PSNR (GT) ↑ | 15.73 | 15.89 | 15.52 | **15.86** |
| Boundary LPIPS ↓ | 0.1024 | 0.1016 | 0.1054 | **0.0988** |
| Face LPIPS ↓ | 0.8040 | 0.8049 | **0.8026** | 0.8047 |
| ArcFace Cos ↑ | 0.3239 | 0.3232 | **0.3325** | 0.3224 |

> Hair FID / Boundary FID: 샘플 수 부족(107장)으로 측정 불가

---

## 4. 분석

**back_only vs none**: PSNR(+0.15), SSIM(+0.002), Edge IoU(+0.001), Boundary LPIPS(-0.001) 에서 소폭 개선. 그러나 LPIPS(GT)는 none이 우세. 전반적으로 **차이가 미미하다**.

**주목할 점**:
- `front_only`가 Chamfer, SSIM, PSNR, Boundary LPIPS에서 오히려 가장 좋음
- `all`이 Sketch LPIPS, ArcFace에서 최고 — 전역 제약이 스케치 충실도에는 유리할 수 있음
- 네 variant 간 절대 수치 차이가 매우 작아 통계적 유의성 불명확

**해석**: 기존 matte conditioning(MatteCNN + matte_latent)이 이미 공간 정보를 충분히 인코딩하고 있어, residual gate의 추가 효과가 제한적으로 나타났다. 이는 설계 단계부터 인지된 리스크 시나리오에 해당한다.

→ **Contribution 무게중심을 Curriculum Learning (ablation_cl) SOTA 결과로 이동** 권장. 본 ablation은 "input conditioning만으로 충분함을 실험으로 확인"이라는 보조 evidence로 활용.

---

## 5. 정성 결과 (braid test set, 5장)

> 열 순서: GT Sketch | none | back_only | all | front_only

### braid_2534
| GT Sketch | none | back_only | all | front_only |
|:---------:|:----:|:---------:|:---:|:----------:|
| <img src="dataset/braid/sketch/test/braid_2534.png" width="150"> | <img src="none-braid/braid_2534_hairpatch.png" width="150"> | <img src="back_only/braid_2534_hairpatch.png" width="150"> | <img src="all/braid_2534_hairpatch.png" width="150"> | <img src="front_only/braid_2534_hairpatch.png" width="150"> |

### braid_2537
| GT Sketch | none | back_only | all | front_only |
|:---------:|:----:|:---------:|:---:|:----------:|
| <img src="dataset/braid/sketch/test/braid_2537.png" width="150"> | <img src="none-braid/braid_2537_hairpatch.png" width="150"> | <img src="back_only/braid_2537_hairpatch.png" width="150"> | <img src="all/braid_2537_hairpatch.png" width="150"> | <img src="front_only/braid_2537_hairpatch.png" width="150"> |

### braid_2539
| GT Sketch | none | back_only | all | front_only |
|:---------:|:----:|:---------:|:---:|:----------:|
| <img src="dataset/braid/sketch/test/braid_2539.png" width="150"> | <img src="none-braid/braid_2539_hairpatch.png" width="150"> | <img src="back_only/braid_2539_hairpatch.png" width="150"> | <img src="all/braid_2539_hairpatch.png" width="150"> | <img src="front_only/braid_2539_hairpatch.png" width="150"> |

### braid_2548
| GT Sketch | none | back_only | all | front_only |
|:---------:|:----:|:---------:|:---:|:----------:|
| <img src="dataset/braid/sketch/test/braid_2548.png" width="150"> | <img src="none-braid/braid_2548_hairpatch.png" width="150"> | <img src="back_only/braid_2548_hairpatch.png" width="150"> | <img src="all/braid_2548_hairpatch.png" width="150"> | <img src="front_only/braid_2548_hairpatch.png" width="150"> |

### braid_2562
| GT Sketch | none | back_only | all | front_only |
|:---------:|:----:|:---------:|:---:|:----------:|
| <img src="dataset/braid/sketch/test/braid_2562.png" width="150"> | <img src="none-braid/braid_2562_hairpatch.png" width="150"> | <img src="back_only/braid_2562_hairpatch.png" width="150"> | <img src="all/braid_2562_hairpatch.png" width="150"> | <img src="front_only/braid_2562_hairpatch.png" width="150"> |
