# Back-12 ControlNet (raw matte) 평가 보고

## 실험 설정

| | block_offset | matte conditioning | ctrl_cond |
|:---:|:---:|:---:|:---:|
| **Back12-raw** | **12** (back blocks 12~23) | raw matte only (`zero_matte_feat=true`) | `cat([sketch_lat, matte_raw], 17ch)` |
| **17ch** (baseline) | 0 (front blocks 0~11) | MatteCNN + raw | `cat([sketch_lat + matte_feat, matte_raw], 17ch)` |

- 두 모델 모두 SD3.5-medium 기반 12-block ControlNet, unbraid 40ep pretrain → braid 40ep finetune.
- 평가: `--split combined` (braid + unbraid test 합산), `eval_metrics.py`.

> ⚠️ **주의 — 비교 시 교란요인 2가지**
> Back12-raw vs 17ch는 (1) **블록 위치**(back vs front)와 (2) **matte 조건**(raw-only vs CNN+raw)이 **동시에** 다름. 따라서 아래 차이를 "back block 때문" 하나로 귀속할 수 없음. 순수 block 위치 효과를 보려면 `back12_cnn_raw`(back + CNN+raw)와의 비교가 필요.
> 또한 17ch는 평가 당시 `pytorch_fid` 미설치로 **Hair/Boundary FID가 N/A** — FID는 두 모델 간 직접 비교 불가.

---

Epoch 40 avg loss: 0.0444

## 평가 결과 (braid 40ep, combined)

| Metric | Back12-raw | 17ch (front12) | 우열 |
|--------|:---:|:---:|:---:|
| Edge IoU ↑ | 0.0629 | **0.0635** | 17ch 근소 |
| Chamfer Dist ↓ | 4.9823 | **4.7625** | 17ch |
| Sketch LPIPS ↓ | 0.7160 | **0.7066** | 17ch |
| Hair FID ↓ | 0.8443 | N/A | — |
| LPIPS (GT) ↓ | 0.3017 | **0.2751** | 17ch |
| SSIM (GT) ↑ | 0.6132 | **0.6139** | 동등 |
| PSNR (GT) ↑ | 11.5386 | **12.2753** | 17ch |
| Boundary FID ↓ | 0.0007 | N/A | — |
| Boundary LPIPS ↓ | **0.0055** | 0.0056 | 동등 |
| Face LPIPS ↓ | **0.0005** | 0.0006 | 동등(back 근소) |
| ArcFace Cos ↑ | 0.7088 | **0.7273** | 17ch |

(**굵게** = 더 우수)

---

## 분석

**1. Sketch Fidelity (구조 추종)** — 17ch 우위
Edge IoU, Chamfer, Sketch LPIPS 모두 17ch가 앞섬. 특히 Chamfer 4.98 vs 4.76으로 back12-raw의 스케치 윤곽 추종이 약간 무딤. 두 절댓값 모두 braid 50ep 수준(mcs ~4.7대)과 비슷한 대역이라 큰 격차는 아님.

**2. Generation Quality (생성 품질)** — 17ch 우위가 가장 뚜렷
PSNR 11.54 vs 12.28 (−0.74dB), LPIPS(GT) 0.302 vs 0.275로 17ch가 명확히 더 GT에 가까운 픽셀/지각 품질. SSIM은 0.613으로 사실상 동률. → **back-block + raw-only 조합이 hair 영역 복원력에서 손해**를 봄.

**3. Boundary / Identity (경계·얼굴)** — 사실상 동등
Boundary LPIPS, Face LPIPS는 소수점 4자리에서만 갈리는 수준으로 차이 없음. ArcFace Cos만 17ch(0.727)가 back12-raw(0.709)보다 약간 높아 얼굴 보존이 조금 더 좋음. 경계/얼굴은 matte로 영역이 강제되는 부분이라 두 방식 모두 안정적.

**4. 종합**
모든 품질 지표에서 **17ch(front-block, CNN+raw)가 동등하거나 우위**, 열세 항목 없음. 다만 격차는 PSNR을 빼면 대부분 작음. Back-block ControlNet으로 옮기면서 동시에 MatteCNN을 제거한 것이 품질 저하의 원인일 수 있으나, 두 요인이 섞여 있어 단정 불가.

---

## Back12-raw vs mcs1 vs mcs3 (matte 조건 비교)

- **mcs1**: front12, matte cond ✓ (raw + MatteCNN), gate ✗
- **mcs3**: front12, matte cond ✗ (sketch만), gate ✗
- **Back12-raw**: back12, raw matte only
- 셋 다 unbraid 40ep → braid **40ep**, combined 평가. (mcs1/mcs3 수치는 `[0606]matte_cnn.md` 재평가본)

| Metric | Back12-raw | mcs1 | mcs3 |
|--------|:---:|:---:|:---:|
| Edge IoU ↑ | 0.0629 | **0.0640** | 0.0628 |
| Chamfer Dist ↓ | 4.9823 | **4.7013** | 4.8089 |
| Sketch LPIPS ↓ | **0.7160** | 0.7599 | 0.7567 |
| Hair FID ↓ | **0.8443** | 1.1078 | 1.5352 |
| LPIPS (GT) ↓ | 0.3017 | **0.2973** | 0.3066 |
| SSIM (GT) ↑ | **0.6132** | 0.5893 | 0.5866 |
| PSNR (GT) ↑ | 11.5386 | **11.6486** | 11.5611 |
| Boundary FID ↓ † | **0.0007** | 0.0121 | 0.0108 |
| Boundary LPIPS ↓ † | **0.0055** | 0.0197 | 0.0220 |
| Face LPIPS ↓ † | **0.0005** | 0.0033 | 0.0033 |
| ArcFace Cos ↑ | **0.7088** | 0.6860 | 0.7014 |

(**굵게** = 3개 중 최고)

> ⚠️ **† 행은 공정 비교 불가**
> Boundary FID / Boundary LPIPS / Face LPIPS에서 Back12-raw(0.0007 / 0.0055 / 0.0005)와 mcs1·mcs3(0.01~0.02 / 0.02 / 0.0033)의 **스케일이 15~20배 차이**. mcs5/mcs6는 Back12-raw와 같은 소(小)스케일인데 mcs1/mcs3만 큼 → 모델 차이가 아니라 **평가/합성 조건(face 합성 방식 등) 차이**로 추정됨. 이 세 행에서 Back12가 "이긴" 것은 의미 없음. 공정 비교하려면 mcs1/mcs3를 Back12와 동일 조건으로 재평가해야 함.

### 분석 (비교 가능한 지표 기준)
- **Sketch Fidelity**: Edge IoU·Chamfer는 **mcs1**(front + 전체 matte)이 최고. 단 Sketch LPIPS는 Back12-raw가 더 낮음.
- **Generation Quality**: PSNR·LPIPS(GT)는 **mcs1**, SSIM·Hair FID는 **Back12-raw**가 우위 → 혼재. mcs3(matte 無)는 대체로 셋 중 최하위라 **matte 조건이 있는 게 낫다**는 점은 일관됨.
- **Identity (ArcFace)**: Back12-raw 0.7088로 최고지만 mcs3(0.7014)와 근소.
- **종합**: mcs1(front + full matte)과 Back12-raw(back + raw matte)가 **항목별로 엎치락뒤치락**하며 뚜렷한 승자 없음. matte 조건을 아예 뺀 mcs3가 가장 약함.

## 다음 단계 (제언)
- **back12_cnn_raw**(back + CNN+raw) 평가로 *블록 위치* 단일 효과 분리.
- 17ch를 `pytorch_fid` 설치 후 **재평가**해 Hair/Boundary FID 직접 비교 확보.
- FID 절댓값(Hair 0.84, Boundary 0.0007)은 `compute_fid`가 `dims=64`로 계산한 값이라 일반적 FID 스케일과 다름 — 동일 설정 간 상대비교로만 해석.
