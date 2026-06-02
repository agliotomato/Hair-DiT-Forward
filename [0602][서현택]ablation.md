## Matte 역할 분리 Ablation

Matte는 두 가지 경로로 작동할 수 있다:

- **Matte cond.**: `ctrl_cond`에 matte_feat + raw_matte를 포함시켜 ControlNet 입력으로 넣는 것
- **Matte gate**: residual을 transformer에 주입할 때 token-level matte mask로 게이팅하는 것

|                    | Matte gate ✗ | Matte gate ✓ |
|--------------------|:------------:|:------------:|
| **Matte cond. ✓**  | Current (A)  |  실험 1 (B)  |
| **Matte cond. ✗**  |  실험 3 (D)  |  실험 2 (C)  |

### 각 조건

| Tag | ctrl_cond | matte_gate | 비고 |
|-----|-----------|:----------:|------|
| A (current) | sketch + matte (17ch) | ✗ | 기준선, in-distribution |
| B | sketch + matte (17ch) | ✓ | 주입 단계만 OOD |
| C | sketch only (zero matte, 17ch) | ✓ | ctrl_cond + 주입 모두 OOD |
| D | sketch only (zero matte, 17ch) | ✗ | ctrl_cond만 OOD |

### 주의

- A만 완전히 in-distribution (학습 시 matte_gate 없음)
- B, C, D는 모두 inference-time ablation (재학습 없음)
- 재학습 없는 ablation이므로 논문에서 "inference-time component ablation"으로 명시 필요

### 예상 민감 지표

1. **Boundary LPIPS** — matte gate 효과 가장 잘 드러남
2. **Face LPIPS / ArcFace** — matte conditioning 없을 때 배경 침범 여부
3. **Hair LPIPS(GT), SSIM** — 전반적 생성 품질
