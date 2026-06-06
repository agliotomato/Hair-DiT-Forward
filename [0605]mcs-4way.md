# MCS 4-way Ablation 결과 보고


## 4-way Ablation

|     | gate ✗  | gate ✓  |
|:------:|:------:|:------:|
| **matte cond ✓** | **mcs1** | **mcs2**  |
| **matte cond ✗** | **mcs3** | **mcs4**  |

- **matte cond ✓** : matte + stetch 모두 사용 
- **matte cond ✗** : sketch만 사용

---
## 이미지 비교
unbraid 40epoch 학습 후 braid **40epoch** 학습 결과
| Sketch | GT | mcs1 | mcs2 | mcs3 | mcs4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](results/report/sketch_test__braid_2534.png) | ![](results/report/img_test__braid_2534.png) | ![](results/report/mcs1_braid__braid_2534.png) | ![](results/report/mcs2_braid__braid_2534.png) | ![](results/report/mcs3_braid__braid_2534.png) | ![](results/report/mcs4_braid__braid_2534.png) |
| ![](results/report/sketch_test__braid_2537.png) | ![](results/report/img_test__braid_2537.png) | ![](results/report/mcs1_braid__braid_2537.png) | ![](results/report/mcs2_braid__braid_2537.png) | ![](results/report/mcs3_braid__braid_2537.png) | ![](results/report/mcs4_braid__braid_2537.png) |
| ![](results/report/sketch_test__braid_2539.png) | ![](results/report/img_test__braid_2539.png) | ![](results/report/mcs1_braid__braid_2539.png) | ![](results/report/mcs2_braid__braid_2539.png) | ![](results/report/mcs3_braid__braid_2539.png) | ![](results/report/mcs4_braid__braid_2539.png) |

unbraid 40epoch 학습 후 braid **50epoch** 학습 결과
| Sketch | GT | mcs1 | mcs2 | mcs3 | mcs4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](results/report/sketch_test__braid_2534.png) | ![](results/report/img_test__braid_2534.png) | ![](results/report/mcs1_braid_50__braid_2534.png) | ![](results/report/mcs2_braid_50__braid_2534.png) | ![](results/report/mcs3_braid_50__braid_2534.png) | ![](results/report/mcs4_braid_50__braid_2534.png) |
| ![](results/report/sketch_test__braid_2537.png) | ![](results/report/img_test__braid_2537.png) | ![](results/report/mcs1_braid_50__braid_2537.png) | ![](results/report/mcs2_braid_50__braid_2537.png) | ![](results/report/mcs3_braid_50__braid_2537.png) | ![](results/report/mcs4_braid_50__braid_2537.png) |
| ![](results/report/sketch_test__braid_2539.png) | ![](results/report/img_test__braid_2539.png) | ![](results/report/mcs1_braid_50__braid_2539.png) | ![](results/report/mcs2_braid_50__braid_2539.png) | ![](results/report/mcs3_braid_50__braid_2539.png) | ![](results/report/mcs4_braid_50__braid_2539.png) |


## 평가 결과

unbraid 40epoch 학습 후 braid **40epoch** 평가 결과
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

unbraid 40epoch 학습 후 braid **50epoch** 평가 결과
| Metric | mcs1 | mcs2 | mcs3 | mcs4 |
|--------|------|------|------|------|
| Edge IoU ↑ | 0.0639 | 0.0641 | 0.0632 | 0.0626 |
| Chamfer Dist ↓ | 4.6947 | 4.6660 | 4.7713 | 4.8808 |
| Sketch LPIPS ↓ | 0.7611 | 0.7704 | 0.7617 | 0.7368 |
| Hair FID ↓ | 1.0666 | 1.5530 | 1.5522 | 1.9780 |
| LPIPS (GT) ↓ | 0.2941 | 0.3292 | 0.3057 | 0.3266 |
| SSIM (GT) ↑ | 0.5899 | 0.5888 | 0.5862 | 0.5918 |
| PSNR (GT) ↑ | 11.7388 | 11.4236 | 11.6754 | 11.4673 |
| Boundary FID ↓ | 0.0108 | 0.0162 | 0.0108 | 0.0206 |
| Boundary LPIPS ↓ | 0.0193 | 0.0212 | 0.0213 | 0.0227 |
| Face LPIPS ↓ | 0.0032 | 0.0037 | 0.0032 | 0.0037 |
| ArcFace Cos ↑ | 0.6877 | 0.6682 | 0.6979 | 0.6638 |
