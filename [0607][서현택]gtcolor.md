# GT-color 평가 결과 (mcs1~6)

_생성일: 2026-06-07 · 데이터: `gtcolor/` 재추론 결과 (GT recolor sketch 입력, num_steps=20) · braid 107 + unbraid 466 = 573_

> **보고 규칙**: paired 지표(Sketch LPIPS·Edge IoU·LPIPS-GT·Boundary LPIPS)는 split별 값으로 분리 보고, **FID 3종(Hair·Boundary·Full)은 전체 통합 573 기준**(split 간 동일 값).

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


## ① braid 전용 (n=107)

| 지표 (방향) | 축 | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 | best |
|---|---|---|---|---|---|---|---|---|
| Sketch LPIPS ↓ | 구조 | 0.7335 | 0.7369 | **0.7056** | 0.7111 | 0.7309 | 0.7234 | mcs3 |
| Edge IoU ↑ | 구조(보조) | 0.1020 | **0.1030** | 0.1012 | 0.1010 | 0.1003 | 0.1019 | mcs2 |
| Hair FID ↓ | 리얼리즘 | 47.26 | 44.23 | **38.05** | 40.64 | 50.10 | 42.88 | mcs3 |
| LPIPS (GT) ↓ | 외형 | 0.1430 | 0.1532 | **0.1412** | 0.1494 | 0.1490 | 0.1437 | mcs3 |
| Boundary FID ↓ | 경계 | 1.5930 | 1.6936 | **1.4981** | 1.6823 | 1.6203 | 1.6670 | mcs3 |
| Boundary LPIPS ↓ | 경계(보조) | **0.0028** | 0.0030 | 0.0028 | 0.0035 | 0.0028 | 0.0029 | mcs1 |
| Full-portrait FID ↓ | 전역 조화 | 18.56 | 17.28 | 17.18 | 17.37 | 18.67 | **17.05** | mcs6 |


---

## ② unbraid 전용 (n=466)

| 지표 (방향) | 축 | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 | best |
|---|---|---|---|---|---|---|---|---|
| Sketch LPIPS ↓ | 구조 | 0.6803 | 0.6855 | **0.6555** | 0.6650 | 0.6759 | 0.6705 | mcs3 |
| Edge IoU ↑ | 구조(보조) | 0.0526 | 0.0525 | 0.0518 | **0.0529** | 0.0521 | 0.0524 | mcs4 |
| Hair FID ↓ | 리얼리즘 | 47.26 | 44.23 | **38.05** | 40.64 | 50.10 | 42.88 | mcs3 |
| LPIPS (GT) ↓ | 외형 | 0.1527 | 0.1546 | 0.1535 | 0.1552 | 0.1517 | **0.1507** | mcs6 |
| Boundary FID ↓ | 경계 | 1.5930 | 1.6936 | **1.4981** | 1.6823 | 1.6203 | 1.6670 | mcs3 |
| Boundary LPIPS ↓ | 경계(보조) | **0.0035** | 0.0037 | 0.0035 | 0.0039 | 0.0035 | 0.0035 | mcs1 |
| Full-portrait FID ↓ | 전역 조화 | 18.56 | 17.28 | 17.18 | 17.37 | 18.67 | **17.05** | mcs6 |



---

## ③ macro = (braid + unbraid) / 2

| 지표 (방향) | 축 | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 | best |
|---|---|---|---|---|---|---|---|---|
| Sketch LPIPS ↓ | 구조 | 0.7069 | 0.7112 | **0.6805** | 0.6881 | 0.7034 | 0.6969 | mcs3 |
| Edge IoU ↑ | 구조(보조) | 0.0773 | **0.0777** | 0.0765 | 0.0769 | 0.0762 | 0.0771 | mcs2 |
| Hair FID ↓ | 리얼리즘 | 47.26 | 44.23 | **38.05** | 40.64 | 50.10 | 42.88 | mcs3 |
| LPIPS (GT) ↓ | 외형 | 0.1478 | 0.1539 | 0.1473 | 0.1523 | 0.1504 | **0.1472** | mcs6 |
| Boundary FID ↓ | 경계 | 1.5930 | 1.6936 | **1.4981** | 1.6823 | 1.6203 | 1.6670 | mcs3 |
| Boundary LPIPS ↓ | 경계(보조) | **0.0031** | 0.0034 | 0.0031 | 0.0037 | 0.0031 | 0.0032 | mcs1 |
| Full-portrait FID ↓ | 전역 조화 | 18.56 | 17.28 | 17.18 | 17.37 | 18.67 | **17.05** | mcs6 |


---

## ④ 유의차 확정 (paired test, combined n=573)

paired t-test + Wilcoxon signed-rank. mean diff = A−B, **±는 차이의 95% CI**, **유의 = 두 검정 모두 p<0.05**.
per-image 짝이 있는 4지표만 대상 (FID 3종은 분포 단위라 검정 불가). 검정 원본: `gtcolor_stats.md`.

**① Sketch LPIPS (구조) — 유일하게 강하게 유의한 분리축**

| 비교 (A vs B) | mean diff (A−B) | 95% CI | p | 결론 |
|---|--:|--:|--:|---|
| mcs3 vs mcs1 | −0.0254 | ±0.0042 | <1e-4 | **mcs3 유의 승** |
| mcs6 vs mcs1 | −0.0099 | ±0.0031 | <1e-4 | **mcs6 유의 승** |
| mcs3 vs mcs6 | −0.0155 | ±0.0039 | <1e-4 | **mcs3 유의 승** |

→ **mcs3 < mcs6 < mcs1** 순서가 세 쌍 모두 p<1e-4로 확정 (CI가 0 미포함). 구조 충실도는 실재하는 차이.

**② LPIPS-GT (외형) — mcs6만 유의하게 우세**

| 비교 (A vs B) | mean diff (A−B) | 95% CI | p | 결론 |
|---|--:|--:|--:|---|
| mcs6 vs mcs1 | −0.0015 | ±0.0010 | 0.0057 | **mcs6 유의 승** |
| mcs6 vs mcs3 | −0.0017 | ±0.0013 | 0.0072 | **mcs6 유의 승** |
| mcs1 vs mcs3 | −0.0003 | ±0.0013 | 0.66 | n.s. (동률) |

→ GT 외형 재현은 **mcs6가 mcs1·mcs3보다 유의하게 우세**, **mcs1 ≈ mcs3는 통계적 동률**.

**③ Edge IoU · ④ Boundary LPIPS — 유의차 없음 수준**

- 차이가 ≤0.0008 / ≈0 수준이고 t·Wilcoxon 결과가 엇갈리거나 n.s. → **의미 있는 차이로 보기 어려움**.

> **요약**: paired 4지표 중 통계적으로 의미 있는 분리는 **Sketch LPIPS(→mcs3)** 와 **LPIPS-GT(→mcs6)** 둘뿐. 나머지는 노이즈 수준. mcs1은 Boundary LPIPS에서 수치상 1위지만 차이가 n.s.라 **유의하게 우세한 축은 없음** (구조는 검정 대상 {mcs1·3·6} 중 최하, 외형은 mcs3와 동률).

---

## 종합 결론

- 세 기준(braid·unbraid·macro) **모두 mcs3가 1위** — 순위가 split에 거의 흔들리지 않음.
- **mcs3**: 구조(Sketch LPIPS, **유의**)·리얼리즘(Hair FID 38.05)·경계(Boundary FID) 핵심 축을 석권. braid에서 7지표 중 4개 1위.
- **mcs6**: 세 기준 모두 2위. **GT 외형 정합(LPIPS-GT)은 mcs6가 유의하게 best** + 전역 조화(Full FID)가 강점.
- **mcs5**: 대체로 최하위 (Hair FID 50.10, Full FID 18.67 둘 다 꼴찌).
- **유의차 단서**: paired 검정상 의미 있는 차이는 **Sketch LPIPS(→mcs3)** 와 **LPIPS-GT(→mcs6)** 뿐. mcs3가 종합 best인 결정타는 **구조(유의) + Hair/Boundary FID 격차**이며, **GT 외형만 보면 mcs6가 유의하게 우세**하다는 트레이드오프가 존재.
