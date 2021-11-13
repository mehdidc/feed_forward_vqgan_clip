import os
import json
import torch.nn as nn
from modules.cloob.clip.model import CLIPGeneral
import torch
from torch.nn import functional as F
import torch
class CLOOB(nn.Module):
    def __init__(self, path="cloob_rn50_yfcc_epoch_28.pt", resize=False):
        super().__init__()
        checkpoint = torch.load(path, map_location='cpu')
        model_config_file = os.path.join("modules", "cloob", "training", "model_configs", checkpoint['model_config_file'])
        method = checkpoint['method']
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        model_info['method'] = method
        model = CLIPGeneral(**model_info)
        sd = checkpoint["state_dict"]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        self.model = model
        self.resize = resize

    def encode_text(self, x):
        return self.model.encode_text(x)

    def encode_image(self, x):
        h = self.model.encode_image(x)
        return h
