## BLD(Blended Latent Diffusion) 적용 방식별 비교

src_latent를 결과물에 반영하는 세 가지 방식을 비교한다.

1. **BLD (매 스텝 블렌딩)**: 디노이징 매 스텝마다 matte 바깥 영역을 src의 noised latent로 갈아끼움
2. **합성만 (composite_full)**: 디노이징 중에는 개입하지 않고, 최종 hair_latent를 디코딩하기 전에 src_latent와 한 번만 합성
3. **미적용 (hair patch only)**: src_latent를 전혀 사용하지 않고 생성된 hair_latent만 디코딩

| 구분 | mcs1 | mcs2 | mcs3 | mcs4 | mcs5 | mcs6 |
|---|---|---|---|---|---|---|
| ① BLD (매 스텝 블렌딩) | ![mcs1 BLD](results/report/mcs1_bld__braid_2534.png) | ![mcs2 BLD](results/report/mcs2_bld__braid_2534.png) | ![mcs3 BLD](results/report/mcs3_bld__braid_2534.png) | ![mcs4 BLD](results/report/mcs4_bld__braid_2534.png) | ![mcs5 BLD](results/report/mcs5_bld__braid_2534.png) | ![mcs6 BLD](results/report/mcs6_bld__braid_2534.png) |
| ② 합성만 (composite_full) | ![mcs1 합성만](results/report/mcs1_composite_only__braid_2534.png) | ![mcs2 합성만](results/report/mcs2_composite_only__braid_2534.png) | ![mcs3 합성만](results/report/mcs3_composite_only__braid_2534.png) | ![mcs4 합성만](results/report/mcs4_composite_only__braid_2534.png) | ![mcs5 합성만](results/report/mcs5_composite_only__braid_2534.png) | ![mcs6 합성만](results/report/mcs6_composite_only__braid_2534.png) |
| ③ 미적용 (hair patch only) | ![mcs1 미적용](results/report/mcs1_patch_only__braid_2534.png) | ![mcs2 미적용](results/report/mcs2_patch_only__braid_2534.png) | ![mcs3 미적용](results/report/mcs3_patch_only__braid_2534.png) | ![mcs4 미적용](results/report/mcs4_patch_only__braid_2534.png) | ![mcs5 미적용](results/report/mcs5_patch_only__braid_2534.png) | ![mcs6 미적용](results/report/mcs6_patch_only__braid_2534.png) |
