"""
HairControlNet: SD3.5 ControlNet for hair region generation.

Architecture (full-image / raw-matte spec):
  ctrl_cond = cat(sketch_latent[16ch], matte_downsampled[1ch])  → 17ch total
  - No MatteCNN. Matte is downsampled via bilinear interpolation only.
  - 12 ControlNet blocks, sequential warm-start from transformer blocks 0..11
    (SD3ControlNetModel.from_transformer default; block i ← transformer block i).
    pos_embed_input is 17ch, zero-initialized (zero-conv).

Forward:
  inputs: noisy_latent (B,16,64,64), sketch (B,3,512,512), matte (B,1,512,512), sigmas (B,)
  1. sketch → frozen VAE encode → sketch_latent (B,16,64,64)
  2. matte  → bilinear downsample → matte_latent (B,1,64,64)
  3. ctrl_cond = cat([sketch_latent, matte_latent], dim=1)  (B,17,64,64)
  4. SD3ControlNetModel forward → 12 block_samples
  5. return block_samples, null_encoder_hs (expanded to B), null_pooled (expanded to B)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SD3ControlNetModel, SD3Transformer2DModel

from src.models.vae_wrapper import VAEWrapper


class SketchDecoderHead(nn.Module):
    """
    Lightweight decoder: 12 ControlNet block_samples → sketch (B, 3, 512, 512).

    block_samples shape: list of (B, seq_len, 1152)  — SD3 sequence format.
    Image tokens are the first spatial_size² entries (32×32=1024 for 512px input).
    """

    def __init__(self, num_blocks: int = 12, hidden_dim: int = 1152, image_tokens: int = 1024):
        super().__init__()
        self.spatial_size = int(image_tokens ** 0.5)  # 32
        self.num_tokens   = image_tokens

        self.block_weights = nn.Parameter(torch.zeros(num_blocks))

        # 32×32 → 512×512 via 4× bilinear upsample
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 1),
            nn.GroupNorm(32, 256), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, block_samples: list[torch.Tensor]) -> torch.Tensor:
        weights = self.block_weights.softmax(dim=0)
        agg = sum(
            w * b[:, : self.num_tokens, :]
            for w, b in zip(weights, block_samples)
        )  # (B, image_tokens, hidden_dim)

        B = agg.shape[0]
        S = self.spatial_size
        spatial = agg.reshape(B, S, S, -1).permute(0, 3, 1, 2).to(dtype=torch.float32)
        return self.decoder(spatial).to(dtype=agg.dtype)


class HairControlNet(nn.Module):
    """
    SD3.5 ControlNet for hair synthesis.

    Trainable: controlnet (SD3ControlNetModel), null text embeddings.
    Frozen:    vae (VAEWrapper, passed in).
    """

    NULL_ENC_SHAPE    = (1, 333, 4096)
    NULL_POOLED_SHAPE = (1, 2048)

    def __init__(
        self,
        model_id: str,
        vae: VAEWrapper,
        num_layers: int = 12,
        local_files_only: bool = False,
        use_sketch_decoder: bool = False,
        zero_matte_cond: bool = False,
    ):
        super().__init__()

        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )

        # from_transformer does TWO things we rely on:
        #   - pos_embed_input is 17ch (num_extra_conditioning_channels=1 default:
        #     16ch sketch latent + 1ch matte), zero-initialized (standard zero-conv).
        #   - sequentially warm-starts ControlNet blocks 0..11 from transformer
        #     blocks 0..11 (block i ← block i). This is the init we keep.
        self.controlnet = SD3ControlNetModel.from_transformer(
            transformer,
            num_layers=num_layers,
            load_weights_from_transformer=True,
        )

        del transformer
        torch.cuda.empty_cache()

        self.sketch_decoder: SketchDecoderHead | None = (
            SketchDecoderHead() if use_sketch_decoder else None
        )

        self.null_encoder_hidden_states = nn.Parameter(
            torch.zeros(*self.NULL_ENC_SHAPE)
        )
        self.null_pooled_projections = nn.Parameter(
            torch.zeros(*self.NULL_POOLED_SHAPE)
        )

        self.zero_matte_cond = zero_matte_cond
        self._vae = vae

    def _build_ctrl_cond(
        self,
        condition_image: torch.Tensor,
        matte: torch.Tensor,
        device,
        dtype,
        enable_grad: bool = False,
    ) -> torch.Tensor:
        """cat(image_latent[16ch], matte_downsampled[1ch]) → (B, 17, 64, 64)."""
        if enable_grad:
            img_latent = self._vae.encode_for_grad(
                condition_image.to(dtype=dtype)
            ).to(device=device, dtype=dtype)
        else:
            img_latent = self._vae.encode(
                condition_image.to(dtype=dtype)
            ).to(device=device, dtype=dtype)

        matte_latent = F.interpolate(
            matte.to(device=device, dtype=dtype),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, 64, 64)
        if self.zero_matte_cond:
            matte_latent = torch.zeros_like(matte_latent)

        return torch.cat([img_latent, matte_latent], dim=1)  # (B, 17, 64, 64)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        sketch: torch.Tensor,
        matte: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_latent: (B, 16, 64, 64)
            sketch:       (B, 3, 512, 512) in [0, 1]
            matte:        (B, 1, 512, 512) in [0, 1]
            sigmas:       (B,)
        Returns:
            block_samples:   list[12] of ControlNet residuals
            null_encoder_hs: (B, 333, 4096)
            null_pooled:     (B, 2048)
        """
        B      = noisy_latent.shape[0]
        device = noisy_latent.device
        dtype  = noisy_latent.dtype

        ctrl_cond   = self._build_ctrl_cond(sketch, matte, device, dtype)
        null_enc_hs = self.null_encoder_hidden_states.expand(B, -1, -1).to(device=device, dtype=dtype)
        null_pooled = self.null_pooled_projections.expand(B, -1).to(device=device, dtype=dtype)

        block_samples = self.controlnet(
            hidden_states=noisy_latent,
            controlnet_cond=ctrl_cond,
            encoder_hidden_states=null_enc_hs,
            pooled_projections=null_pooled,
            timestep=sigmas.to(device=device, dtype=dtype),
            return_dict=False,
        )[0]

        return block_samples, null_enc_hs, null_pooled

    def _get_features_impl(
        self,
        condition_image: torch.Tensor,
        matte: torch.Tensor,
        enable_grad: bool = False,
    ) -> list[torch.Tensor]:
        B      = condition_image.shape[0]
        device = condition_image.device
        dtype  = condition_image.dtype

        ctrl_cond   = self._build_ctrl_cond(condition_image, matte, device, dtype, enable_grad)
        null_enc_hs = self.null_encoder_hidden_states.expand(B, -1, -1).to(device=device, dtype=dtype)
        null_pooled = self.null_pooled_projections.expand(B, -1).to(device=device, dtype=dtype)

        dummy_latent = torch.zeros(B, 16, 64, 64, device=device, dtype=dtype)
        dummy_sigma  = torch.zeros(B, device=device, dtype=dtype)

        return self.controlnet(
            hidden_states=dummy_latent,
            controlnet_cond=ctrl_cond,
            encoder_hidden_states=null_enc_hs,
            pooled_projections=null_pooled,
            timestep=dummy_sigma,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def get_features(self, sketch: torch.Tensor, matte: torch.Tensor) -> list[torch.Tensor]:
        return self._get_features_impl(sketch, matte)
