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

| stem | mcs1<br>sk+CNN+raw | mcs2<br>sk+CNN+raw+gate | mcs3<br>sk only | mcs4<br>sk+gate | mcs5<br>sk+raw | mcs6<br>sk+CNN |
|------|------|------|------|------|------|------|
| braid_2534 | ![](cross-id/braid/mcs1_original_sketch_braid_2534.png) | ![](cross-id/braid/mcs2_original_sketch_braid_2534.png) | ![](cross-id/braid/mcs3_original_sketch_braid_2534.png) | ![](cross-id/braid/mcs4_original_sketch_braid_2534.png) | ![](cross-id/braid/mcs5_original_sketch_braid_2534.png) | ![](cross-id/braid/mcs6_original_sketch_braid_2534.png) |
| braid_2562 | ![](cross-id/braid/mcs1_original_sketch_braid_2562.png) | ![](cross-id/braid/mcs2_original_sketch_braid_2562.png) | ![](cross-id/braid/mcs3_original_sketch_braid_2562.png) | ![](cross-id/braid/mcs4_original_sketch_braid_2562.png) | ![](cross-id/braid/mcs5_original_sketch_braid_2562.png) | ![](cross-id/braid/mcs6_original_sketch_braid_2562.png) |
| braid_2572 | ![](cross-id/braid/mcs1_original_sketch_braid_2572.png) | ![](cross-id/braid/mcs2_original_sketch_braid_2572.png) | ![](cross-id/braid/mcs3_original_sketch_braid_2572.png) | ![](cross-id/braid/mcs4_original_sketch_braid_2572.png) | ![](cross-id/braid/mcs5_original_sketch_braid_2572.png) | ![](cross-id/braid/mcs6_original_sketch_braid_2572.png) |
| CM_1004 | ![](cross-id/unbraid/mcs1_original_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs2_original_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs3_original_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs4_original_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs5_original_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs6_original_sketch_CM_1004.png) |
| CM_1006 | ![](cross-id/unbraid/mcs1_original_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs2_original_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs3_original_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs4_original_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs5_original_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs6_original_sketch_CM_1006.png) |

---

## gt_sketch (①②③④ 평가)

| stem | mcs1<br>sk+CNN+raw | mcs2<br>sk+CNN+raw+gate | mcs3<br>sk only | mcs4<br>sk+gate | mcs5<br>sk+raw | mcs6<br>sk+CNN |
|------|------|------|------|------|------|------|
| braid_2534 | ![](cross-id/braid/mcs1_gt_sketch_braid_2534.png) | ![](cross-id/braid/mcs2_gt_sketch_braid_2534.png) | ![](cross-id/braid/mcs3_gt_sketch_braid_2534.png) | ![](cross-id/braid/mcs4_gt_sketch_braid_2534.png) | ![](cross-id/braid/mcs5_gt_sketch_braid_2534.png) | ![](cross-id/braid/mcs6_gt_sketch_braid_2534.png) |
| braid_2562 | ![](cross-id/braid/mcs1_gt_sketch_braid_2562.png) | ![](cross-id/braid/mcs2_gt_sketch_braid_2562.png) | ![](cross-id/braid/mcs3_gt_sketch_braid_2562.png) | ![](cross-id/braid/mcs4_gt_sketch_braid_2562.png) | ![](cross-id/braid/mcs5_gt_sketch_braid_2562.png) | ![](cross-id/braid/mcs6_gt_sketch_braid_2562.png) |
| braid_2572 | ![](cross-id/braid/mcs1_gt_sketch_braid_2572.png) | ![](cross-id/braid/mcs2_gt_sketch_braid_2572.png) | ![](cross-id/braid/mcs3_gt_sketch_braid_2572.png) | ![](cross-id/braid/mcs4_gt_sketch_braid_2572.png) | ![](cross-id/braid/mcs5_gt_sketch_braid_2572.png) | ![](cross-id/braid/mcs6_gt_sketch_braid_2572.png) |
| CM_1004 | ![](cross-id/unbraid/mcs1_gt_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs2_gt_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs3_gt_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs4_gt_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs5_gt_sketch_CM_1004.png) | ![](cross-id/unbraid/mcs6_gt_sketch_CM_1004.png) |
| CM_1006 | ![](cross-id/unbraid/mcs1_gt_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs2_gt_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs3_gt_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs4_gt_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs5_gt_sketch_CM_1006.png) | ![](cross-id/unbraid/mcs6_gt_sketch_CM_1006.png) |
