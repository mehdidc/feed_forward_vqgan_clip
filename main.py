from clize import run
import random
import argparse
import math
from pathlib import Path
import sys
import torchvision
sys.path.insert(1, 'taming-transformers')
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import taming.modules 
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import clip
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image
import stylegan
from mlp_mixer_pytorch import Mixer
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply

def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)

def synth(model, z):
    # if args.vqgan_checkpoint == 'vqgan_openimages_f16_8192.ckpt':
        # z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    # else:
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    x  = clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)
    return x

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
            # K.RandomHorizontalFlip(p=0.5),
            # K.RandomVerticalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            # K.RandomSharpness(0.3,p=0.4),
            # K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5),
            # K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5),
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
            
)
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):

            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            # offsetx = torch.randint(0, sideX - size + 1, ())
            # offsety = torch.randint(0, sideY - size + 1, ())
            # cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            # cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            # cutout = transforms.Resize(size=(self.cut_size, self.cut_size))(input)
            
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([len(batch), 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def train():
    # texts = ["sunflower"]
    texts = [t.strip() for t in open("input.txt").readlines()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqgan_config = "vqgan_imagenet_f16_16384.yaml"
    vqgan_checkpoint = "vqgan_imagenet_f16_16384.ckpt"
    clip_model = "ViT-B/32"
    lr = 0.001
    epochs = 1000000
    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    # net = stylegan.SGDummy(latent_dim=512, image_size=16, truncate=32, nb_channels=256, network_capacity=16, depth=1).to(device)
    clip_dim = 512
    channels = 256
    nb_gen = 1
    vq = 16
    net = Mixer(input_dim=512, image_size=vq, channels=channels, patch_size=1, dim=128, depth=8, nb_gen=nb_gen).to(device)

    opt = optim.Adam(net.parameters(), lr=lr)
    mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1,-1,1,1).to(device)
    std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1,-1,1,1).to(device)
    clip_size = 224
    cutn = 8
    make_cutouts = MakeCutouts(clip_size, cutn=cutn)
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    L = 1. 
    bs = 2
    step = 0
    div_loss_coef = 0
    mask = (1-torch.eye(nb_gen)).view(1,1,nb_gen,nb_gen,1).to(device)
    for e in range(epochs):
        random.shuffle(texts)
        for j in range(0, len(texts), bs):
            T = texts[j:j+bs]
            bs = len(T)
            #bs,clip_dim
            H = perceptor.encode_text(clip.tokenize(T).to(device)).float()
            z = net(H)
            #bs*nb_gen, channels, vq, vq
            z = z.contiguous()
            z = z.view(bs * nb_gen, channels, vq, vq)
            z = clamp_with_grad(z, z_min.min(), z_max.max())
            #bs*nb_gen, 3, h, w
            xr = synth(model, z)
            #cutn*bs*nb_gen,3,h,w
            x = make_cutouts(xr)
            x = (x-mean)/std
            #cutn*bs*nb_gen,clip_dim
            embed = perceptor.encode_image(x).float()
            #nb_gen*bs,clip_dim
            H = H.repeat(nb_gen, 1)
            #cutn*nb_gen*bs,clip_dim
            H = H.repeat(cutn, 1)
            H = H.view(cutn, nb_gen, bs, clip_dim)
            #cutn,bs,nb_gen,clip_dim
            # H = H.permute(0,2,1,3)
            H = F.normalize(H, dim=-1)
            #cutn*bs*nb_gen,clip_dim
            H = H.view(-1, clip_dim)

            embed = F.normalize(embed, dim=1)

            if div_loss_coef:
                em = embed.view(cutn,bs,nb_gen,clip_dim)
                em_a = em.view(cutn,bs,nb_gen,1,clip_dim)
                em_b = em.view(cutn,bs,1,nb_gen,clip_dim)
                NB = mask.sum()
                # div_loss = (mask*((em_a*em_b).sum(dim=-1))).sum() / (NB*bs*cutn)#minimize dot product
                # div_loss = -(mask*((em_a-em_b).norm(dim=-1).div(2).arcsin().pow(2).mul(2))).sum() / (NB*bs*cutn)#minimize dot product
                div_loss = -z.view(bs,nb_gen,channels,vq,vq).std(dim=1).mean()
            else:
                div_loss = torch.Tensor([0]).float().to(device)
            #maximize clip score
            dists = (H.sub(embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean()
            opt.zero_grad()
            loss = dists + div_loss * div_loss_coef
            loss.backward()
            opt.step()
            L = loss.item() * 0.1 + L * 0.9 
            if step % 100 == 0:
                print(e, step, L, loss.item(), dists.item(), div_loss.item())
                g = torchvision.utils.make_grid(xr.cpu(), nrow=nb_gen)
                TF.to_pil_image(g).save('progress.png')
                TF.to_pil_image(g).save(f'steps/progress_{step:010d}.png')
                torch.save(net, "model.th")
                with open("progress.txt", "w") as fd:
                    fd.write("\n".join(T))
            step += 1


def test(model_path, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.load(model_path, map_location="cpu").to(device)
    vqgan_config = "vqgan_imagenet_f16_16384.yaml"
    vqgan_checkpoint = "vqgan_imagenet_f16_16384.ckpt"
    clip_model = "ViT-B/32"
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    texts = text.split("|")
    H = perceptor.encode_text(clip.tokenize(texts).to(device)).float()
    z = net(H)
    z = clamp_with_grad(z, z_min.min(), z_max.max())
    xr = synth(model, z)
    g = torchvision.utils.make_grid(xr.cpu(), nrow=len(xr))
    TF.to_pil_image(g).save('gen.png')

if __name__ == "__main__":
    run([train, test])
