"""아키텍처 검증: dual_attention_layers, 블록 복사 커버리지, 주입 매핑, 17ch 입력."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from diffusers import SD3Transformer2DModel
from src.models.controlnet_sd35 import HairControlNet
from src.models.vae_wrapper import VAEWrapper

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

tf = SD3Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True)
dual = getattr(tf.config, "dual_attention_layers", None)
print(f"num transformer blocks : {len(tf.transformer_blocks)}")
print(f"dual_attention_layers  : {dual}")

# 각 controlnet block i ← transformer 2i 의 구조 일치 여부.
# 불일치해도 fresh-init 아님: sequential(from_transformer) 값이 dual-extra에 남음.
print("\nblock copy coverage (controlnet i <- transformer 2i):")
for i in range(12):
    cn_dual = i in (dual or [])
    tf_dual = (2 * i) in (dual or [])
    if cn_dual == tf_dual:
        mark = "full copy from block 2i"
    else:
        mark = "attn1/ff from 2i; attn2/norm1-extra retain block i (sequential)"
    print(f"  cn{i:2d} <- tf{2*i:2d} | cn_dual={cn_dual!s:5} tf_dual={tf_dual!s:5} | {mark}")

# 주입 매핑
n_cn, n_tf = 12, len(tf.transformer_blocks)
interval = n_tf / n_cn
print(f"\ninjection map (residual -> transformer block), interval={interval}:")
mapping = {}
for idx in range(n_tf):
    r = int(idx / interval)
    mapping.setdefault(r, []).append(idx)
for r in range(n_cn):
    print(f"  residual[{r:2d}] -> blocks {mapping.get(r)}")

# 17ch 입력 + forward smoke test
vae = VAEWrapper.from_pretrained(model_id=MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
cn = HairControlNet(model_id=MODEL_ID, vae=vae, num_layers=12, local_files_only=True).cuda().to(torch.bfloat16).eval()
print(f"\npos_embed_input in_channels: {cn.controlnet.pos_embed_input.proj.in_channels} (expect 17)")

with torch.no_grad():
    nl = torch.randn(1, 16, 64, 64, device="cuda", dtype=torch.bfloat16)
    sk = torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.bfloat16)
    mt = torch.rand(1, 1, 512, 512, device="cuda", dtype=torch.bfloat16)
    sg = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
    bs, _, _ = cn(noisy_latent=nl, sketch=sk, matte=mt, sigmas=sg)
print(f"num residuals: {len(bs)} (expect 12), residual[0] shape: {tuple(bs[0].shape)}")
print("\nOK")
