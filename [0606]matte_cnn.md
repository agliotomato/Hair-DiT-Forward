# MatteCNN Ablation 결과 보고 (mcs5,6)

## 실험 설정

| | raw matte (1ch) | MatteCNN feat |
|:------:|:------:|:------:|
| **mcs1** | ✓ | ✓ |
| **mcs3** | ✗ | ✗ |
| **mcs5** | ✓ | ✗ |
| **mcs6** | ✗ | ✓ |



---

## 이미지 비교

unbraid 40epoch 학습 후 braid **40epoch** 학습 결과

| Sketch | GT | mcs1 | mcs3 | mcs5 | mcs6 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](results/report/sketch_test__braid_2534.png) | ![](results/report/img_test__braid_2534.png) | ![](results/report/mcs1_braid__braid_2534.png) | ![](results/report/mcs3_braid__braid_2534.png) | ![](results/report/mcs5_braid_2534.png) | ![](results/report/mcs6_braid_2534.png) |
| ![](results/report/sketch_test__braid_2537.png) | ![](results/report/img_test__braid_2537.png) | ![](results/report/mcs1_braid__braid_2537.png) | ![](results/report/mcs3_braid__braid_2537.png) | ![](results/report/mcs5_braid_2537.png) | ![](results/report/mcs6_braid_2537.png) |
| ![](results/report/sketch_test__braid_2539.png) | ![](results/report/img_test__braid_2539.png) | ![](results/report/mcs1_braid__braid_2539.png) | ![](results/report/mcs3_braid__braid_2539.png) | ![](results/report/mcs5_braid_2539.png) | ![](results/report/mcs6_braid_2539.png) |

---

## 평가 결과

unbraid 40epoch 학습 후 braid **40epoch** 평가 결과

| Metric | mcs1 | mcs3 | mcs5 | mcs6 |
|--------|------|------|------|------|
| Edge IoU ↑ | 0.1040 | 0.1020 | 0.0637 | 0.0638 |
| Chamfer Dist ↓ | 2.7142 | 2.7612 | 4.6665 | 4.7403 |
| Sketch LPIPS ↓ | 0.7884 | 0.7612 | 0.7584 | 0.7386 |
| Hair FID ↓ | 1.0006 | 1.3923 | 1.3141 | 1.1242 |
| LPIPS (GT) ↓ | 0.3199 | 0.3255 | 0.2904 | 0.2850 |
| SSIM (GT) ↑ | 0.6004 | 0.5976 | 0.6075 | 0.6116 |
| PSNR (GT) ↑ | 11.9526 | 11.6444 | 12.0964 | 12.2796 |
| Boundary FID ↓ | 0.0074 | 0.0094 | 0.0006 | 0.0006 |
| Boundary LPIPS ↓ | 0.0158 | 0.0195 | 0.0049 | 0.0048 |
| Face LPIPS ↓ | 0.0014 | 0.0014 | 0.0005 | 0.0005 |
| ArcFace Cos ↑ | 0.7870 | 0.7907 | 0.7000 | 0.7258 |
