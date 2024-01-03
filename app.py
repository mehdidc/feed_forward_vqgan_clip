import torch
import gradio as gr
import numpy as np
import clip
import main
from omegaconf import OmegaConf
from main import *

from download_weights import PRIOR_MODEL
MODELS = [
    "cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.1.th",
    "cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.2.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.2.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.3.th",
    "cc12m_32x1024_mlp_mixer_cloob_rn50_256x256_v0.3.th",
    "cc12m_256x16_xtransformer_clip_ViTB32_512x512_v0.3.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_pixelrecons_256x256_v0.4.th",
    "cc12m_32x1024_mlp_mixer_openclip_laion2b_ViTB32_256x256_v0.4.th",
    "cc12m_32x1024_mlp_mixer_openclip_laion2b_imgEmb_ViTB32_256x256_v0.4.th",
    "cc12m_1x1024_mlp_mixer_openclip_laion2b_ViTB32_512x512_v0.4.th",
]

DEFAULT_MODEL = "cc12m_32x1024_mlp_mixer_openclip_laion2b_ViTB32_256x256_v0.4.th"
GRID_SIZES = [
    "1x1",
    "2x2",
    "4x4",
    "8x8",
]

def download_and_load_model(model_path):
    download_from_path(model_path)
    return load_model(model_path)

def download_and_load_prior_model(model_path):
    download_from_path(model_path)
    return load_prior_model(model_path)

def download_from_path(path):
    from download_weights import download, model_url
    download(model_url[path], target=path)

class Predictor:

    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nets = {
            model_path: download_and_load_model(model_path).cpu()
            for model_path in MODELS
        }
        self.priors = {}
        self.perceptors = {}
        self.vqgans = {}
        for path, net in self.nets.items():
            config = net.config
            vqgan_config = config.vqgan_config
            vqgan_checkpoint = config.vqgan_checkpoint
            clip_model = config.clip_model
            clip_model_path = config.get("clip_model_path")
            # Load CLIP model if not already done 
            if (clip_model, clip_model_path) not in self.perceptors:
                perceptor = load_clip_model(clip_model, path=clip_model_path).eval().requires_grad_(False).to(self.device)
                self.perceptors[(clip_model, clip_model_path)] = perceptor
            # Load VQGAN model if not already done
            if (vqgan_config, vqgan_checkpoint) not in self.vqgans:
                model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(self.device)
                z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
                z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
                self.vqgans[(vqgan_config, vqgan_checkpoint)] = model, z_min, z_max
            # Load PRIOR model if not already done
            if PRIOR_MODEL[path] not in self.priors:
                self.priors[PRIOR_MODEL[path]] = download_and_load_prior_model(PRIOR_MODEL[path]).cpu()

    def predict(
        self, 
        prompt, 
        model, 
        prior,
        grid_size, 
        seed
    ):
        torch.manual_seed(seed)
        random.seed(seed)
        if model == "random":
            model = random.choice(list(self.nets.keys()))
        net = self.nets[model]

        net.to(self.device)

        config = net.config
        clip_model = config.clip_model
        clip_model_path = config.get("clip_model_path")
        vqgan_config = config.vqgan_config
        vqgan_checkpoint = config.vqgan_checkpoint
        grid_size_h, grid_size_w = grid_size.split("x")
        grid_size_h = int(grid_size_h)
        grid_size_w = int(grid_size_w)
        toks = clip.tokenize([prompt], truncate=True)
        perceptor = self.perceptors[(clip_model, clip_model_path)]
        vqgan, z_min, z_max = self.vqgans[(vqgan_config, vqgan_checkpoint)]
        if prior:
            prior_model = self.priors[PRIOR_MODEL[model]]
            prior_model.to(self.device)
        with torch.no_grad():
            H = perceptor.encode_text(toks.to(self.device)).float()
            H = H.repeat(grid_size_h*grid_size_w, 1)
            if prior:
                H = H.view(len(H), -1, 1, 1)
                H = prior_model.sample(H)
                H = H.view(len(H), -1)
            z = net(H)
            z = clamp_with_grad(z, z_min.min(), z_max.max())
            xr = synth(vqgan, z)
        grid = torchvision.utils.make_grid(xr.cpu(), nrow=grid_size_h)
        img = torchvision.transforms.functional.to_pil_image(grid)
        return img

if __name__ == "__main__":
    pred = Predictor()
    pred.setup()
    def fn(prompt:str, model:str, prior:bool, grid_size:str, seed:int):
        return pred.predict(prompt, model, prior ,grid_size, seed)
    iface = gr.Interface(fn=fn, inputs=["text", gr.Dropdown(MODELS,value=DEFAULT_MODEL), "checkbox", gr.Dropdown(GRID_SIZES, value="1x1"), "slider"], outputs="image")
    iface.launch()

