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
| Edge IoU ↑ | 0.1040 | 0.1046 | 0.1020 | 0.1033 |
| Chamfer Dist ↓ | 2.7142 | 2.6657 | 2.7612 | 2.7521 |
| Sketch LPIPS ↓ | 0.7884 | 0.7805 | 0.7612 | 0.7551 |
| Hair FID ↓ | 1.0006 | 1.3858 | 1.3923 | 1.5115 |
| LPIPS (GT) ↓ | 0.3199 | 0.3626 | 0.3255 | 0.3548 |
| SSIM (GT) ↑ | 0.6004 | 0.6018 | 0.5976 | 0.5990 |
| PSNR (GT) ↑ | 11.9526 | 11.5631 | 11.6444 | 11.5737 |
| Boundary FID ↓ | 0.0074 | 0.0119 | 0.0094 | 0.0135 |
| Boundary LPIPS ↓ | 0.0158 | 0.0192 | 0.0195 | 0.0205 |
| Face LPIPS ↓ | 0.0014 | 0.0016 | 0.0014 | 0.0017 |
| ArcFace Cos ↑ | 0.7870 | 0.7786 | 0.7907 | 0.7807 |

unbraid 40epoch 학습 후 braid **50epoch** 평가 결과
| Metric | mcs1 | mcs2 | mcs3 | mcs4 |
|--------|------|------|------|------|
| Edge IoU ↑ | 0.0639 | x | 0.0632 | x |
| Chamfer Dist ↓ | 4.6947 | x | 4.7713 | x |
| Sketch LPIPS ↓ | 0.7611 | x | 0.7617 | x |
| Hair FID ↓ | N/A | x | N/A | x |
| LPIPS (GT) ↓ | 0.2941 | x | 0.3057 | x |
| SSIM (GT) ↑ | 0.5899 | x | 0.5862 | x |
| PSNR (GT) ↑ | 11.7388 | x | 11.6754 | x |
| Boundary FID ↓ | N/A | x | N/A | x |
| Boundary LPIPS ↓ | 0.0193 | x | 0.0213 | x |
| Face LPIPS ↓ | 0.0032 | x | 0.0032 | x |
| ArcFace Cos ↑ | 0.6877 | x | 0.6979 | x |

> mc2, mc4는 추후 기재 예정