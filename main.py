"""
Feed forward VQGAN-CLIP model, where the goal is to eliminate the need for optimizing the latent
space of VQGAN for each input prompt.

- The training code is heavily based on the VQGAN-CLIP notebook <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>, thanks
to all the authors who contributed to the notebook (@crowsonkb, @advadnoun, @Eleiber, @Crimeacs, @Abulafia)
- Thanks to @lucidrains, the MLP mixer model (`mlp_mixer_pytorch.py`)  is from <https://github.com/lucidrains/mlp-mixer-pytorch>.
- Thanks to Taming Transformers authors <https://github.com/CompVis/taming-transformers>, the code uses VQGAN pre-trained model and
VGG16 feature space perceptual loss <https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py>
- Thanks to @afiaka87, who provided the blog captions dataset for experimentation.
"""
import os
from clize import run
from glob import glob
import random
import math
from pathlib import Path
import sys
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from PIL import ImageFile, Image
import numpy as np
import kornia.augmentation as K
import imageio

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.tensorboard import SummaryWriter

from taming.models import cond_transformer, vqgan
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.lpips import normalize_tensor
import taming.modules 

import clip
from clip import simple_tokenizer
from mlp_mixer_pytorch import Mixer


from omegaconf import OmegaConf

try:
    import horovod.torch as hvd
    USE_HOROVOD = True
except ImportError:
    USE_HOROVOD = False

decode = simple_tokenizer.SimpleTokenizer().decode
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

def encode_images(fname, *, root_folder="", out="features.pkl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    paths = open(fname).readlines()
    paths = [root_folder + p.strip() for p in paths]
    features = []
    for p in paths:
        print(p)
        image = preprocess(Image.open(p)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        features.append(image_features.cpu())
    features = torch.cat(features)
    torch.save(features, out)


def tokenize(path, out="tokenized.pkl", max_length:int=None):
    """
    tokenize and save to a pkl file

    path: str
        can be either a text file where each line is a text prompt
        or a glob pattern where each file is a text prompt
    out: str
        output pkl file
    max_length: int
        this can be used to filter text prompts and retain only
        ones that are up to `max_length`
    """
    if "*" in paths:
        texts = [open(f).read().strip() for f in glob(paths)]
    else:
        texts = [l.strip() for l in open(paths).readlines()]
        if max_length:
            texts = [text for text in texts if len(text) <= max_length]
    T = clip.tokenize(texts)
    torch.save(T, out)

def train(config_file):

    config = OmegaConf.load(config_file)
    if not hasattr(config, "folder"):
        config.folder = os.path.dirname(config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if USE_HOROVOD:
        hvd.init()
        if device == "cuda":
            torch.cuda.set_device(hvd.local_rank())

    lpips = LPIPS()
    lpips.load_from_pretrained()
    lpips = lpips.to(device)
    
    # path can be the following:
    # - a path to a text file where each line is a text prompt
    # - a glob pattern (*) of text files where each file is a text prompt
    # - a pkl file created using `tokenize`, where each text prompt is already tokenized
    path = config.path
    if path.endswith("pkl"):
        toks = torch.load(path)
    elif "*" in path:
        texts = [open(f).read().strip() for f in glob(path)]
        toks = clip.tokenize(texts)
    else:
        texts = [t.strip() for t in open(path).readlines()]
        toks = clip.tokenize(texts)
    print(f"Number of text prompts:{len(toks)}")

    vqgan_config = config.vqgan_config
    vqgan_checkpoint = config.vqgan_checkpoint
    clip_model = config.clip_model 
    lr = config.lr
    epochs = config.epochs

    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    clip_dim = 512
    clip_size = 224
    vq_channels = 256
    vq_image_size = config.vq_image_size if config.vq_image_size else 16
    noise_dim = config.noise_dim
    
    model_path = os.path.join(config.folder, "model.th")
    
    if os.path.exists(model_path):
        print(f"Resuming from {model_path}")
        net = torch.load(model_path, map_location="cpu")
    else:
        net = Mixer(
            input_dim=clip_dim+noise_dim, 
            image_size=vq_image_size, 
            channels=vq_channels, 
            patch_size=1, 
            dim=config.dim, 
            depth=config.depth, 
            dropout=config.dropout)
    net = net.to(device)
    net.config = config
    opt = optim.Adam(net.parameters(), lr=lr)
    
    rank_zero =  (USE_HOROVOD and hvd.rank() == 0) or not USE_HOROVOD
    if rank_zero:
        log_writer = SummaryWriter(config.folder)
    else:
        log_writer = None

    if USE_HOROVOD:
        opt = hvd.DistributedOptimizer(opt)
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(opt, root_rank=0)

    mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1,-1,1,1).to(device)
    std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1,-1,1,1).to(device)
    cutn = config.cutn
    make_cutouts = MakeCutouts(clip_size, cutn=cutn)
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    bs = config.batch_size
    repeat = config.repeat
    nb_noise = config.nb_noise
    dataset = torch.utils.data.TensorDataset(toks)
    if not config.diversity_mode:
        config.diversity_mode = "between_same_prompts"

    if USE_HOROVOD:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=hvd.size(),
            rank=hvd.rank(),
        )
        shuffle=  None
    else:
        sampler = None
        shuffle = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=1, sampler=sampler, shuffle=shuffle)
    if nb_noise:
        if hasattr(net, "NOISE"):
            NOISE = net.NOISE
        else:
            NOISE = torch.randn(nb_noise,noise_dim)
        if USE_HOROVOD:
            NOISE = hvd.broadcast(NOISE, root_rank=0)
        net.NOISE = NOISE

    avg_loss = 1. 
    step = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for T, in dataloader:
            T = T.to(device)
            bs = len(T)
            if T.dtype == torch.long:
                #bs,clip_dim
                H = perceptor.encode_text(T).float()
            else:
                H = T.float()
            #repeat*bs,clip_dim
            H = H.repeat(repeat, 1)
            if noise_dim:
                inds = np.arange(len(NOISE))
                np.random.shuffle(inds)
                inds = inds[:repeat]
                noise = NOISE[inds].to(device).repeat(bs, 1).view(bs,repeat,-1).permute(1,0,2).contiguous().view(bs*repeat,-1)
                Hi = torch.cat((H,noise),dim=1)
            else:
                Hi = H
            z = net(Hi)
            #bs, vq_channels, vq_image_size, vq_image_size
            z = z.contiguous()
            z = z.view(repeat*bs, vq_channels, vq_image_size, vq_image_size)
            z = clamp_with_grad(z, z_min.min(), z_max.max())
            #bs, 3, h, w
            xr = synth(model, z)
            if config.diversity_coef:
                div = 0
                # for feats in [z]:
                for feats in lpips.net( (xr-mean)/std):
                    feats = normalize_tensor(feats)
                    _, cc,hh,ww = feats.shape
                    if config.diversity_mode == "between_same_prompts":
                        div += ( (feats.view(repeat, 1, bs, cc,hh,ww) - feats.view(1, repeat, bs, cc,hh,ww)) ** 2).sum(dim=3).mean()
                    elif config.diversity_mode == "all":
                        nb = len(feats)
                        div += ( (feats.view(nb, 1, cc,hh,ww) - feats.view(1, nb, cc,hh,ww)) ** 2).sum(dim=2).mean()
                    else:
                        raise ValueError("diversity_mode should be 'between_same_prompts' lr 'all'")
            else:
                div = torch.Tensor([0.]).to(device)
            #cutn*bs,3,h,w
            x = make_cutouts(xr)
            x = (x-mean)/std
            #cutn*bs,clip_dim
            embed = perceptor.encode_image(x).float() # generated image features
            #cutn*bs,clip_dim
            H = H.repeat(cutn, 1)
            H = H.view(cutn, repeat, bs, clip_dim)
            H = F.normalize(H, dim=-1)
            #cutn*bs,clip_dim
            H = H.view(-1, clip_dim)
            
            #cutn*bs,clip_dim
            embed = F.normalize(embed, dim=1)
            
            #dist between prompt features `H` and generated image features `embed`
            dists = (H.sub(embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean()
            opt.zero_grad()

            # 1) minimize distance between generated images CLIP features and text prompt features
            # 2) maximize diversity between generated images with the same prompt
            loss = dists  - config.diversity_coef * div
            loss.backward()
            opt.step()
            if rank_zero: 
                log_writer.add_scalar("loss", loss.item(), step)
                log_writer.add_scalar("dists", dists.item(), step)
                log_writer.add_scalar("diversity", div.item(), step)
            avg_loss = loss.item() * 0.01 + avg_loss * 0.99 
            if rank_zero and step % config.log_interval == 0:
                print(epoch, step, avg_loss, loss.item(), dists.item(), div.item())
                grid = torchvision.utils.make_grid(xr.cpu(), nrow=bs)
                TF.to_pil_image(grid).save(os.path.join(config.folder, 'progress.png'))
                TF.to_pil_image(grid).save(os.path.join(config.folder, f'progress_{step:010d}.png'))
                torch.save(net, model_path)
                if T.dtype == torch.long:
                    text = "\n".join([decode(t.tolist()) for t in T])
                    with open(os.path.join(config.folder, "progress.txt"), "w") as fd:
                        fd.write(text)
                    with open(os.path.join(config.folder, f"progress_{step:010d}.txt"), "w") as fd:
                        fd.write(text)
            step += 1


def test(model_path, text, *, nb_repeats=1, out_path="gen.png"):
    """
    generated an image or a set of images from a model given a text prompt

    model_path: str
        path of the model
    text: str
        can either be:
         - a text prompt. several text prompts can be provided  by delimiting them using "|"
         - a path to a text file .txt, where each line is a text prompt
    nb_repeats: int
        number of times the same text prompt is repeated
        with different noise vectors
    out_path: str
        output path
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.load(model_path, map_location="cpu").to(device)
    config = net.config
    vqgan_config = config.vqgan_config 
    vqgan_checkpoint = config.vqgan_checkpoint
    clip_model = config.clip_model
    clip_dim = 512
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if text.endswith(".txt"):
        texts = [t.strip() for t in open(text).readlines()]
    else:
        texts = text.split("|")
    H = perceptor.encode_text(clip.tokenize(texts).to(device)).float()
    H = H.repeat(nb_repeats, 1)
    noise_dim = net.input_dim - clip_dim
    if noise_dim:
        if hasattr(net, "NOISE"):
            noise = net.NOISE[:nb_repeats].to(device)
            print(noise)
        else:
            noise = torch.randn(len(H), noise_dim).to(device)
        H = torch.cat((H, noise), dim=1)
    with torch.no_grad():
        z = net(H)
        z = clamp_with_grad(z, z_min.min(), z_max.max())
        xr = synth(model, z)
    grid = torchvision.utils.make_grid(xr.cpu(), nrow=nb_repeats)
    TF.to_pil_image(grid).save(out_path)

if __name__ == "__main__":
    run([train, test, tokenize, encode_images])
