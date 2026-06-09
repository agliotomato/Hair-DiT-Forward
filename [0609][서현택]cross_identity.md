# Cross-Identity 생성 결과 비교

대표 이미지 5개 × mcs1~6 비교  
- braid 3개: braid_2534, braid_2562, braid_2572  
- unbraid 2개: CM_1004, CM_1006

---

## MCS 구성

| exp | MatteCNN (16ch) | raw matte (1ch) | gate | ctrl_cond |
|-----|:---:|:---:|:---:|-----------|
| mcs1 | ✅ | ✅ | ❌ | sketch + CNN + raw (full MCS) |
| mcs2 | ✅ | ✅ | ✅ | sketch + CNN + raw + gate |
| mcs3 | ❌ | ❌ | ❌ | sketch only |
| mcs4 | ❌ | ❌ | ✅ | sketch only + gate |
| mcs5 | ❌ | ✅ | ❌ | sketch + raw (CNN 없음) |
| mcs6 | ✅ | ❌ | ❌ | sketch + CNN (raw 없음) |

---

## original_sketch (①②③ 평가)

| sketch | mcs1<br>sk+CNN+raw | mcs2<br>sk+CNN+raw+gate | mcs3<br>sk only | mcs4<br>sk+gate | mcs5<br>sk+raw | mcs6<br>sk+CNN |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| <img src="cross-id/braid/sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs1_original_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs2_original_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs3_original_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs4_original_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs5_original_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs6_original_sketch_braid_2534.png" width="120"> |
| <img src="cross-id/braid/sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs1_original_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs2_original_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs3_original_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs4_original_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs5_original_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs6_original_sketch_braid_2562.png" width="120"> |
| <img src="cross-id/braid/sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs1_original_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs2_original_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs3_original_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs4_original_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs5_original_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs6_original_sketch_braid_2572.png" width="120"> |
| <img src="cross-id/unbraid/sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs1_original_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs2_original_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs3_original_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs4_original_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs5_original_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs6_original_sketch_CM_1004.png" width="120"> |
| <img src="cross-id/unbraid/sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs1_original_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs2_original_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs3_original_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs4_original_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs5_original_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs6_original_sketch_CM_1006.png" width="120"> |

---

## gt_sketch (①②③④ 평가)

| gt_sketch | mcs1<br>sk+CNN+raw | mcs2<br>sk+CNN+raw+gate | mcs3<br>sk only | mcs4<br>sk+gate | mcs5<br>sk+raw | mcs6<br>sk+CNN |
|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|
| <img src="cross-id/braid/gt_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs1_gt_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs2_gt_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs3_gt_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs4_gt_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs5_gt_sketch_braid_2534.png" width="120"> | <img src="cross-id/braid/mcs6_gt_sketch_braid_2534.png" width="120"> |
| <img src="cross-id/braid/gt_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs1_gt_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs2_gt_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs3_gt_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs4_gt_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs5_gt_sketch_braid_2562.png" width="120"> | <img src="cross-id/braid/mcs6_gt_sketch_braid_2562.png" width="120"> |
| <img src="cross-id/braid/gt_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs1_gt_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs2_gt_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs3_gt_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs4_gt_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs5_gt_sketch_braid_2572.png" width="120"> | <img src="cross-id/braid/mcs6_gt_sketch_braid_2572.png" width="120"> |
| <img src="cross-id/unbraid/gt_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs1_gt_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs2_gt_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs3_gt_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs4_gt_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs5_gt_sketch_CM_1004.png" width="120"> | <img src="cross-id/unbraid/mcs6_gt_sketch_CM_1004.png" width="120"> |
| <img src="cross-id/unbraid/gt_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs1_gt_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs2_gt_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs3_gt_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs4_gt_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs5_gt_sketch_CM_1006.png" width="120"> | <img src="cross-id/unbraid/mcs6_gt_sketch_CM_1006.png" width="120"> |


---

## 정량 평가

> **보고 규칙** FID(③)=통합573, 나머지=braid(n=107)/unbraid(n=466)/macro 분리, ±는 95%CI.  
> **bold** = 해당 열 best.

---

### original_sketch — ①②③

#### ① Sketch LPIPS ↓ (구조, vs sketch)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | 0.7618±0.0287 | 0.7695±0.0286 | 0.7279±0.0268 | **0.7081±0.0282** | 0.7673±0.0284 | 0.7580±0.0284 |
| unbraid | 0.7162±0.0101 | 0.7456±0.0099 | 0.7106±0.0100 | **0.6938±0.0090** | 0.7342±0.0099 | 0.7156±0.0107 |
| macro   | 0.7390±0.0152 | 0.7575±0.0151 | 0.7192±0.0143 | **0.7009±0.0148** | 0.7507±0.0151 | 0.7368±0.0152 |

#### ② Edge IoU ↑ (구조 보조, vs sketch)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | 0.1014±0.0037 | **0.1019±0.0037** | 0.0985±0.0033 | 0.0985±0.0034 | 0.1001±0.0036 | 0.1012±0.0036 |
| unbraid | 0.0557±0.0011 | **0.0559±0.0011** | 0.0548±0.0012 | 0.0534±0.0011 | 0.0550±0.0011 | 0.0554±0.0012 |
| macro   | 0.0785±0.0019 | **0.0789±0.0019** | 0.0766±0.0018 | 0.0759±0.0018 | 0.0776±0.0019 | 0.0783±0.0019 |

#### ③ Hair FID ↓ (리얼리즘, 통합 573)

| mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|
| **128.80** | 146.06 | 154.54 | 156.51 | 142.61 | 143.08 |

#### Sketch ΔE ↓ (색상, vs sketch stroke pixels ∩ hair mask, CIE76)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | **58.72±0.82** | 62.96±0.90 | 59.75±0.81 | 63.38±0.84 | 61.75±0.81 | 60.02±0.82 |
| unbraid | **53.14±0.56** | 57.07±0.57 | 53.89±0.51 | 57.07±0.53 | 56.85±0.49 | 53.95±0.53 |
| macro   | **55.93±0.50** | 60.01±0.53 | 56.82±0.48 | 60.23±0.50 | 59.30±0.47 | 56.99±0.48 |

#### PSNR ↑ (화질, vs GT hair mask)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | **11.48±0.26** | 10.98±0.26 | 10.87±0.24 | 10.79±0.26 | 11.16±0.26 | 11.30±0.25 |
| unbraid | **11.19±0.14** | 10.92±0.14 | 10.89±0.13 | 10.82±0.13 | 10.96±0.14 | 11.09±0.13 |
| macro   | **11.34±0.15** | 10.95±0.15 | 10.88±0.14 | 10.80±0.14 | 11.06±0.15 | 11.19±0.14 |

---

### gt_sketch — ①②③④

#### ① Sketch LPIPS ↓ (구조, vs sketch)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | 0.7296±0.0293 | 0.7374±0.0306 | 0.7055±0.0282 | **0.7044±0.0290** | 0.7313±0.0307 | 0.7203±0.0290 |
| unbraid | 0.6660±0.0112 | 0.6723±0.0116 | **0.6453±0.0097** | 0.6513±0.0098 | 0.6607±0.0118 | 0.6522±0.0114 |
| macro   | 0.6978±0.0157 | 0.7048±0.0164 | **0.6754±0.0149** | 0.6779±0.0153 | 0.6960±0.0164 | 0.6862±0.0156 |

#### ② Edge IoU ↑ (구조 보조, vs sketch)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | **0.1014±0.0038** | **0.1014±0.0038** | 0.0997±0.0036 | 0.0991±0.0036 | 0.1008±0.0037 | 0.1004±0.0037 |
| unbraid | 0.0533±0.0014 | 0.0535±0.0014 | 0.0532±0.0012 | **0.0536±0.0012** | 0.0525±0.0014 | 0.0532±0.0014 |
| macro   | 0.0774±0.0020 | **0.0775±0.0020** | 0.0765±0.0019 | 0.0764±0.0019 | 0.0767±0.0020 | 0.0768±0.0020 |

#### ③ Hair FID ↓ (리얼리즘, 통합 573)

| mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|
| 56.44 | 50.44 | **46.47** | 47.29 | 58.66 | 50.57 |

#### ④ LPIPS-GT ↓ (외형, vs GT hair-masked)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | 0.2030±0.0103 | 0.2127±0.0106 | 0.2045±0.0100 | 0.2089±0.0102 | 0.2089±0.0100 | **0.2003±0.0096** |
| unbraid | 0.2171±0.0043 | 0.2201±0.0042 | 0.2238±0.0046 | 0.2225±0.0044 | 0.2172±0.0043 | **0.2155±0.0043** |
| macro   | 0.2100±0.0056 | 0.2164±0.0057 | 0.2141±0.0055 | 0.2157±0.0056 | 0.2130±0.0055 | **0.2079±0.0053** |

#### Sketch ΔE ↓ (색상, vs sketch hair mask, CIE76)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | 50.94±1.92 | **49.35±1.90** | 52.93±1.82 | 52.41±1.98 | 49.70±1.92 | 50.40±1.90 |
| unbraid | 44.03±1.29 | **42.87±1.31** | 47.73±1.18 | 45.52±1.26 | 43.44±1.29 | 43.90±1.27 |
| macro   | 47.49±1.16 | **46.11±1.15** | 50.33±1.09 | 48.97±1.17 | 46.57±1.16 | 47.15±1.14 |


#### PSNR ↑ (화질, vs GT hair mask)

| scope | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| braid   | **14.09±0.25** | 13.98±0.26 | 13.87±0.23 | 13.75±0.25 | 13.94±0.26 | 13.84±0.25 |
| unbraid | **14.64±0.15** | 14.60±0.15 | 14.51±0.13 | 14.60±0.14 | 14.64±0.15 | 14.56±0.16 |
| macro   | **14.37±0.15** | 14.29±0.15 | 14.19±0.13 | 14.17±0.14 | 14.29±0.15 | 14.20±0.15 |

---

### 유의차 검정 (paired t-test + Wilcoxon, α=0.05)

#### original_sketch — mcs1 vs mcs3 / mcs1 vs mcs6

| 지표 | scope | mcs1 vs mcs3 | mcs1 vs mcs6 |
|---|---|---|---|
| Sketch LPIPS ↓ | combined | **mcs3 유의 승** (−0.0109, p<1e-4) | n.s. (+0.0012) |
| Sketch LPIPS ↓ | braid    | **mcs3 유의 승** (−0.0339, p<1e-4) | n.s. (+0.0038) |
| Sketch LPIPS ↓ | unbraid  | **mcs3 유의 승** (−0.0056, p=0.0025) | n.s. (+0.0007) |
| Edge IoU ↑     | combined | **mcs1 유의 승** (+0.0013, p<1e-4) | n.s. (+0.0003) |
| Edge IoU ↑     | braid    | **mcs1 유의 승** (+0.0029, p<1e-4) | n.s. (+0.0001) |
| Edge IoU ↑     | unbraid  | **mcs1 유의 승** (+0.0009, p<1e-4) | n.s. (+0.0003) |
| Sketch ΔE ↓    | combined | **mcs1 유의 승** (−0.81, p<1e-4) | **mcs1 유의 승** (−0.90, p<1e-4) |
| Sketch ΔE ↓    | braid    | **mcs1 유의 승** (−1.03, p=0.0018) | **mcs1 유의 승** (−1.30, p<1e-4) |
| Sketch ΔE ↓    | unbraid  | **mcs1 유의 승** (−0.76, p=0.0003) | **mcs1 유의 승** (−0.81, p<1e-4) |
| PSNR ↑         | combined | **mcs1 유의 승** (+0.36, p<1e-4) | **mcs1 유의 승** (+0.12, p<1e-4) |
| PSNR ↑         | braid    | **mcs1 유의 승** (+0.61, p<1e-4) | **mcs1 유의 승** (+0.18, p=0.0089) |
| PSNR ↑         | unbraid  | **mcs1 유의 승** (+0.30, p<1e-4) | **mcs1 유의 승** (+0.11, p=0.0007) |

#### gt_sketch — mcs1 vs mcs3 / mcs1 vs mcs6

| 지표 | scope | mcs1 vs mcs3 | mcs1 vs mcs6 |
|---|---|---|---|
| Sketch LPIPS ↓ | combined | **mcs3 유의 승** (−0.0213, p<1e-4) | **mcs6 유의 승** (−0.0130, p<1e-4) |
| Sketch LPIPS ↓ | braid    | **mcs3 유의 승** (−0.0241, p<1e-4) | **mcs6 유의 승** (−0.0094, p=0.0058) |
| Sketch LPIPS ↓ | unbraid  | **mcs3 유의 승** (−0.0207, p<1e-4) | **mcs6 유의 승** (−0.0138, p<1e-4) |
| Edge IoU ↑     | combined | n.s. (+0.0004, 한쪽만) | n.s. (+0.0003) |
| Edge IoU ↑     | braid    | **mcs1 유의 승** (+0.0017, p=0.0133) | n.s. (+0.0011) |
| Edge IoU ↑     | unbraid  | n.s. (한쪽만) | n.s. (+0.0002) |
| LPIPS-GT ↓     | combined | **mcs1 유의 승** (−0.0057, p<1e-4) | **mcs6 유의 승** (−0.0018, p=0.0007) |
| LPIPS-GT ↓     | braid    | n.s. (−0.0016) | n.s. (−0.0026) |
| LPIPS-GT ↓     | unbraid  | **mcs1 유의 승** (−0.0067, p<1e-4) | **mcs6 유의 승** (−0.0016, p=0.0027) |
| Sketch ΔE ↓    | combined | **mcs1 유의 승** (−3.38, p<1e-4) | **mcs6 유의 승** (+0.21, p=0.0247) |
| Sketch ΔE ↓    | braid    | **mcs1 유의 승** (−1.99, p<1e-4) | 한쪽만 (+0.54, t=0.0226) |
| Sketch ΔE ↓    | unbraid  | **mcs1 유의 승** (−3.70, p<1e-4) | n.s. (+0.14) |
| ΔE(GT,hair) ↓  | combined | **mcs1 유의 승** (−0.18, p=0.0069) | **mcs1 유의 승** (−0.14, p=0.0099) |
| ΔE(GT,hair) ↓  | braid    | **mcs1 유의 승** (−0.39, p=0.0084) | **mcs1 유의 승** (−0.59, p<1e-4) |
| ΔE(GT,hair) ↓  | unbraid  | n.s. (−0.13) | n.s. (−0.04) |
| PSNR ↑         | combined | **mcs1 유의 승** (+0.15, p<1e-4) | **mcs1 유의 승** (+0.11, p<1e-4) |
| PSNR ↑         | braid    | **mcs1 유의 승** (+0.22, p=0.0010) | **mcs1 유의 승** (+0.25, p<1e-4) |
| PSNR ↑         | unbraid  | **mcs1 유의 승** (+0.13, p<1e-4) | **mcs1 유의 승** (+0.08, p=0.0010) |

---

