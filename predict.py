import cog
import tempfile
from pathlib import Path
import torch
from PIL import Image
import clip
import torchvision
from main import load_vqgan_model, CLIP_DIM, clamp_with_grad, synth

MODELS = [
    "cc12m_32x1024_vitgan_v0.1.th",
    "cc12m_32x1024_vitgan_v0.2.th",
    "cc12m_32x1024_mlp_mixer_v0.2.th",
]
DEFAULT_MODEL = "cc12m_32x1024_vitgan_v0.2.th"

class Predictor(cog.Predictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nets = {
            model_path: torch.load(model_path, map_location="cpu").to(self.device)
            for model_path in MODELS
        }
        config = self.nets[DEFAULT_MODEL].config
        vqgan_config = config.vqgan_config
        vqgan_checkpoint = config.vqgan_checkpoint
        clip_model = config.clip_model
        self.perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(self.device)
        self.model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(self.device)
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    @cog.input("prompt", type=str, help="prompt for generating image")
    @cog.input("model", type=str, default=DEFAULT_MODEL, options=MODELS, help="Model version")
    def predict(self, prompt, model=DEFAULT_MODEL):
        net = self.nets[model]
        toks = clip.tokenize([prompt], truncate=True)
        H = self.perceptor.encode_text(toks.to(self.device)).float()
        with torch.no_grad():
            z = net(H)
            z = clamp_with_grad(z, self.z_min.min(), self.z_max.max())
            xr = synth(self.model, z)
        grid = torchvision.utils.make_grid(xr.cpu(), nrow=len(xr))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        torchvision.transforms.functional.to_pil_image(grid).save(out_path)
        return out_path



