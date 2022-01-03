import cog
import tempfile
from pathlib import Path
import torch
from PIL import Image
import clip
import torchvision
from main import load_vqgan_model, CLIP_DIM, clamp_with_grad, synth, load_clip_model

MODELS = [
    "cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.1.th",
    "cc12m_32x1024_vitgan_clip_ViTB32_256x256_v0.2.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.2.th",
    "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.3.th",
    "cc12m_32x1024_mlp_mixer_cloob_rn50_256x256_v0.3.th",
    "cc12m_256x16_xtransformer_clip_ViTB32_512x512_v0.3.th",
]
DEFAULT_MODEL = "cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.2.th"

class Predictor(cog.Predictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nets = {
            model_path: torch.load(model_path, map_location="cpu").to(self.device)
            for model_path in MODELS
        }
        self.perceptors = {}
        self.vqgans = {}
        for path, net in self.nets.items():
            config = net.config
            vqgan_config = config.vqgan_config
            vqgan_checkpoint = config.vqgan_checkpoint
            clip_model = config.clip_model
            clip_model_path = config.get("clip_model_path")
            # Load CLIP mode if not already done 
            if (clip_model, clip_model_path) not in self.perceptors:
                perceptor = load_clip_model(clip_model, path=clip_model_path).eval().requires_grad_(False).to(self.device)
                self.perceptors[(clip_model, clip_model_path)] = perceptor
            # Load VQGAN model if not already done
            if (vqgan_config, vqgan_checkpoint) not in self.vqgans:
                model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(self.device)
                z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
                z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
                self.vqgans[(vqgan_config, vqgan_checkpoint)] = model, z_min, z_max

    @cog.input("prompt", type=str, help="prompt for generating image")
    @cog.input("model", type=str, default=DEFAULT_MODEL, options=MODELS, help="Model version")
    def predict(self, prompt, model=DEFAULT_MODEL):
        net = self.nets[model]
        config = net.config
        clip_model = config.clip_model
        clip_model_path = config.get("clip_model_path")
        vqgan_config = config.vqgan_config
        vqgan_checkpoint = config.vqgan_checkpoint
        toks = clip.tokenize([prompt], truncate=True)
        perceptor = self.perceptors[(clip_model, clip_model_path)]
        vqgan, z_min, z_max = self.vqgans[(vqgan_config, vqgan_checkpoint)]
        with torch.no_grad():
            H = perceptor.encode_text(toks.to(self.device)).float()
            z = net(H)
            z = clamp_with_grad(z, z_min.min(), z_max.max())
            xr = synth(vqgan, z)
        grid = torchvision.utils.make_grid(xr.cpu(), nrow=len(xr))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        torchvision.transforms.functional.to_pil_image(grid).save(out_path)
        return out_path
