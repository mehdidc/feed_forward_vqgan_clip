import cog
import tempfile
from pathlib import Path
import torch
from PIL import Image
import clip
import torchvision
from main import load_vqgan_model, CLIP_DIM, clamp_with_grad, synth


class Predictor(cog.Predictor):
    def setup(self):
        model_path = "cc12m_32x1024.th"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = torch.load(model_path, map_location="cpu").to(self.device)
        config = self.net.config
        vqgan_config = config.vqgan_config
        vqgan_checkpoint = config.vqgan_checkpoint
        clip_model = config.clip_model
        self.perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(self.device)
        self.model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(self.device)
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    @cog.input("prompt", type=str, help="prompt for generating image")
    def predict(self, prompt):
        toks = clip.tokenize([prompt], truncate=True)
        H = self.perceptor.encode_text(toks.to(self.device)).float()
        with torch.no_grad():
            z = self.net(H)
            z = clamp_with_grad(z, self.z_min.min(), self.z_max.max())
            xr = synth(self.model, z)
        grid = torchvision.utils.make_grid(xr.cpu(), nrow=len(xr))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        torchvision.transforms.functional.to_pil_image(grid).save(out_path)
        return out_path



