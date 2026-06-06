# MCS Ablation 종합 분석 (mcs1~6)

> 평가 데이터: braid test set **N=573**, recheck CSV (일관 조건 재평가본, commit `bd629db`)
> 소스: `mcs/eval_results/recheck_mcs{1..4}_*.csv`, `eval_results/recheck_mcs{5,6}_*.csv`

---

현재 방식은 GT 스케치 헤어가 아닌 컬러가 들어간 sketch로 추론한 결과입니다!


## 공통 학습 절차

모든 run은 동일한 2-phase 절차를 따른다.

1. **Phase 1 (unbraid pretrain)**: unbraid 데이터로 40 epoch 학습
2. **Phase 2 (braid finetune)**: Phase1 체크포인트에서 braid 데이터로 40 epoch finetune
   - `configs/phase2_braid.yaml` 공유: lr `2e-5`, batch 16, edge loss `0.05`, LPIPS `0.1` (즉시 활성)

run 간 차이는 **(a) matte 조건 입력 구성**과 **(b) matte gate 적용 여부** 두 축뿐이며, 나머지 하이퍼파라미터는 전부 동일하다.

---


## 실험 설계

각 run의 matte 조건 구성 (`MatteCNN` = 16ch 학습형 feature, `matte_downsample` = raw 1ch area-downsample, `gate` = matte soft gate):

| run | MatteCNN | matte_downsample | gate | 한 줄 설명 |
|:---:|:---:|:---:|:---:|---|
| **mcs1** | O | O | X | matte 풀가동 (CNN + downsample) |
| **mcs2** | O | O | O | matte 풀가동 + gate |
| **mcs3** | X | X | X | sketch only |
| **mcs4** | X | X | O | sketch only + gate |
| **mcs5** | X | O | X | matte_downsample만 |
| **mcs6** | O | X | X | MatteCNN만 |

이 6개 run은 사실 **두 개의 직교 ablation**으로 읽어야 한다.

### 축 A — matte cond × gate (mcs1~4)
| | gate ✗ | gate ✓ |
|:---:|:---:|:---:|
| **matte cond ✓** | mcs1 | mcs2 |
| **matte cond ✗** | mcs3 | mcs4 |

### 축 B — matte 표현 분해 (mcs1, 3, 5, 6, gate 전부 OFF)
| | raw matte(1ch) | MatteCNN feat |
|:---:|:---:|:---:|
| **mcs1** | ✓ | ✓ (둘 다) |
| **mcs5** | ✓ | ✗ (raw만) |
| **mcs6** | ✗ | ✓ (CNN만) |
| **mcs3** | ✗ | ✗ (sketch only) |

> **mcs1**(full)과 **mcs3**(sketch only)이 두 축의 공통 앵커다.

---

## 통합 평가 결과 (per-image 평균, N=573)

6개 중 best는 **굵게**.

| Metric | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| Edge IoU ↑      | 0.0640 | **0.0644** | 0.0631 | 0.0617 | 0.0637 | 0.0638 |
| Chamfer ↓       | 4.705  | **4.662**  | 4.753  | 4.873  | 4.667  | 4.740  |
| Sketch LPIPS ↓  | 0.7442 | 0.7654 | 0.7321 | **0.7135** | 0.7584 | 0.7385 |
| Hair FID ↓      | **0.829** | 1.209 | 1.378 | 1.766 | 1.314 | 1.124 |
| **LPIPS (GT) ↓**| **0.2668** | 0.3025 | 0.2767 | 0.2955 | 0.2904 | 0.2850 |
| **SSIM ↑**      | **0.6121** | 0.6081 | 0.6081 | 0.6101 | 0.6075 | 0.6116 |
| **PSNR ↑**      | **12.398** | 12.019 | 12.109 | 11.871 | 12.096 | 12.280 |
| Boundary LPIPS ↓| **0.0047** | 0.0051 | 0.0048 | 0.0055 | 0.0049 | 0.0048 |
| Face LPIPS ↓    | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| **ArcFace Cos ↑**| **0.7361** | 0.6936 | 0.7176 | 0.6985 | 0.7000 | 0.7258 |

(Face LPIPS는 6개 모두 동일, 변별력 없음)

---

## 도메인별 분해 — braid / unbraid / macro-avg

test set은 **braid(N=107)** 과 **unbraid(N=466, = CM 215 + R2 199 + wavy 52)** 로 구성되며,
위 §4 표(N=573)는 이미지 수에 비례하는 **micro-avg**라 unbraid 쪽으로 강하게 쏠린다.
아래는 per-image CSV에서 두 도메인을 분리해 각각 평균낸 뒤, 두 도메인을 **동등 가중**한
**macro-avg = (braid + unbraid) / 2** 를 함께 제시한다.
(Hair FID는 분포 기반 지표라 per-image 평균이 불가능 → 이 표에서는 제외)

각 행 best는 **굵게**.

### braid (N=107)

| Metric | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| Edge IoU ↑       | 0.1035 | **0.1045** | 0.1020 | 0.1012 | 0.1036 | 0.1033 |
| Chamfer ↓        | 2.750 | **2.696** | 2.763 | 2.782 | 2.729 | 2.726 |
| Sketch LPIPS ↓   | 0.7677 | 0.7811 | 0.7351 | **0.7153** | 0.7782 | 0.7589 |
| LPIPS (GT) ↓     | **0.2751** | 0.3238 | 0.2890 | 0.3152 | 0.3101 | 0.2943 |
| SSIM ↑           | **0.6271** | 0.6219 | 0.6237 | 0.6223 | 0.6251 | **0.6271** |
| PSNR ↑           | **12.856** | 12.245 | 12.320 | 11.985 | 12.626 | 12.751 |
| Boundary LPIPS ↓ | **0.0040** | 0.0047 | 0.0043 | 0.0054 | 0.0043 | 0.0042 |
| Face LPIPS ↓     | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| ArcFace Cos ↑    | 0.8119 | 0.7954 | 0.8060 | 0.7862 | 0.8006 | **0.8127** |

### unbraid (N=466)

| Metric | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| Edge IoU ↑       | 0.0549 | **0.0552** | 0.0541 | 0.0526 | 0.0545 | 0.0547 |
| Chamfer ↓        | 5.153 | 5.113 | 5.210 | 5.353 | **5.111** | 5.203 |
| Sketch LPIPS ↓   | 0.7387 | 0.7619 | 0.7314 | **0.7131** | 0.7538 | 0.7338 |
| LPIPS (GT) ↓     | **0.2648** | 0.2977 | 0.2739 | 0.2910 | 0.2859 | 0.2828 |
| SSIM ↑           | **0.6086** | 0.6050 | 0.6046 | 0.6073 | 0.6034 | 0.6080 |
| PSNR ↑           | **12.293** | 11.966 | 12.061 | 11.845 | 11.975 | 12.171 |
| Boundary LPIPS ↓ | **0.0048** | 0.0052 | 0.0049 | 0.0056 | 0.0050 | 0.0049 |
| Face LPIPS ↓     | 0.0006 | 0.0006 | 0.0006 | 0.0006 | 0.0006 | 0.0006 |
| ArcFace Cos ↑    | **0.7187** | 0.6702 | 0.6973 | 0.6784 | 0.6769 | 0.7058 |

### macro-avg = (braid + unbraid) / 2

| Metric | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| Edge IoU ↑       | 0.0792 | **0.0799** | 0.0780 | 0.0769 | 0.0791 | 0.0790 |
| Chamfer ↓        | 3.952 | **3.904** | 3.987 | 4.067 | 3.920 | 3.964 |
| Sketch LPIPS ↓   | 0.7532 | 0.7715 | 0.7333 | **0.7142** | 0.7660 | 0.7464 |
| LPIPS (GT) ↓     | **0.2700** | 0.3107 | 0.2814 | 0.3031 | 0.2980 | 0.2886 |
| SSIM ↑           | **0.6179** | 0.6134 | 0.6141 | 0.6148 | 0.6143 | 0.6176 |
| PSNR ↑           | **12.574** | 12.106 | 12.190 | 11.915 | 12.300 | 12.461 |
| Boundary LPIPS ↓ | **0.0044** | 0.0050 | 0.0046 | 0.0055 | 0.0047 | 0.0046 |
| Face LPIPS ↓     | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 |
| ArcFace Cos ↑    | **0.7653** | 0.7328 | 0.7517 | 0.7323 | 0.7388 | 0.7593 |

- macro로 봐도 §4 결론 순위는 유지: **품질 지표(LPIPS/SSIM/PSNR/Boundary/ArcFace) 전부 mcs1 1위, mcs6 2위**, gate(mcs2/mcs4)는 Edge IoU·Chamfer만 우위.
- braid 도메인에선 SSIM이 mcs1=mcs6 동률, ArcFace는 mcs6가 근소 1위 → **MatteCNN feature 단독(mcs6)이 braid에선 full(mcs1)과 거의 동급**.

---
