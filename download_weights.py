import os
from subprocess import call
model_url = {
    "cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.1.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/cc12m_32x1024.th",
    "cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.2.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.2/cc12m_32x1024_vitgan.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.2.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.2/cc12m_32x1024_mlp_mixer.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.3.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.3/cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.3.th",
    "cc12m_32x1024_mlp_mixer_cloob_rn50_256x256_v0.3.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.3/cc12m_32x1024_mlp_mixer_cloob_rn50_256x256_v0.3.th",
    "cc12m_256x16_xtransformer_clip_ViTB32_512x512_v0.3.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.3/cc12m_256x16_xtransformer_clip_ViTB32_512x512_v0.3.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_pixelrecons_256x256_v0.4.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.4/cc12m_32x1024_mlp_mixer_clip_ViTB32_pixelrecons_256x256_v0.4.th",
    "cc12m_32x1024_mlp_mixer_openclip_laion2b_ViTB32_256x256_v0.4.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.4/cc12m_32x1024_mlp_mixer_openclip_laion2b_ViTB32_256x256_v0.4.th",
    "cc12m_32x1024_mlp_mixer_openclip_laion2b_imgEmb_ViTB32_256x256_v0.4.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.4/cc12m_32x1024_mlp_mixer_openclip_laion2b_imgEmb_ViTB32_256x256_v0.4.th",
    "cc12m_1x1024_mlp_mixer_openclip_laion2b_ViTB32_512x512_v0.4.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.4/cc12m_1x1024_mlp_mixer_openclip_laion2b_ViTB32_512x512_v0.4.th",
    "prior_cc12m_2x1024_openclip_laion2b_ViTB32_v0.4.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.4/prior_cc12m_2x1024_openclip_laion2b_ViTB32_v0.4.th",
    "prior_cc12m_2x1024_clip_ViTB32_v0.4.th": "https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.4/prior_cc12m_2x1024_clip_ViTB32_v0.4.th"
}
PRIOR_MODEL = {
    "cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.1.th": "prior_cc12m_2x1024_clip_ViTB32_v0.4.th",
    "cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.2.th": "prior_cc12m_2x1024_clip_ViTB32_v0.4.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.2.th": "prior_cc12m_2x1024_clip_ViTB32_v0.4.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.3.th": "prior_cc12m_2x1024_clip_ViTB32_v0.4.th",
    "cc12m_32x1024_mlp_mixer_cloob_rn50_256x256_v0.3.th": "prior_cc12m_2x1024_clip_ViTB32_v0.4.th",
    "cc12m_256x16_xtransformer_clip_ViTB32_512x512_v0.3.th": "prior_cc12m_2x1024_clip_ViTB32_v0.4.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_pixelrecons_256x256_v0.4.th": "prior_cc12m_2x1024_clip_ViTB32_v0.4.th",
    "cc12m_32x1024_mlp_mixer_openclip_laion2b_ViTB32_256x256_v0.4.th": "prior_cc12m_2x1024_openclip_laion2b_ViTB32_v0.4.th",
    "cc12m_32x1024_mlp_mixer_openclip_laion2b_imgEmb_ViTB32_256x256_v0.4.th": "prior_cc12m_2x1024_openclip_laion2b_ViTB32_v0.4.th",
    "cc12m_1x1024_mlp_mixer_openclip_laion2b_ViTB32_512x512_v0.4.th": "prior_cc12m_2x1024_openclip_laion2b_ViTB32_v0.4.th"
}
def download(url, target=None):
    if target is None:
        target = os.path.basename(url)
    if not os.path.exists(target):
        call(f"wget {url} --output-document={target}", shell=True)
    else:
        print(f"Skipping {target}, already exists")

def download_all():
    download("https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/vqgan_imagenet_f16_16384.yaml")
    download("https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/vqgan_imagenet_f16_16384.ckpt")
    download("https://ml.jku.at/research/CLOOB/downloads/checkpoints/cloob_rn50_yfcc_epoch_28.pt")
    for path, url in model_url.items():
        download(url, path)

if __name__ == "__main__":
    download_all()