#!/bin/sh

wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/vqgan_imagenet_f16_16384.yaml # vqgan_imagenet_f16_16384.yaml
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/vqgan_imagenet_f16_16384.ckpt # vqgan_imagenet_f16_16384.ckpt
wget https://ml.jku.at/research/CLOOB/downloads/checkpoints/cloob_rn50_yfcc_epoch_28.pt
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/cc12m_32x1024.th --output-document=cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.1.th
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.2/cc12m_32x1024_vitgan.th --output-document=cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.2.th
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.2/cc12m_32x1024_mlp_mixer.th --output-document=cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.2.th
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.3/cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.3.th
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.3/cc12m_32x1024_mlp_mixer_cloob_rn50_256x256_v0.3.th
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.3/cc12m_256x16_xtransformer_clip_ViTB32_512x512_v0.3.th
