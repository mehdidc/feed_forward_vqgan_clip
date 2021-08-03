"""
Feed forward VQGAN-CLIP model, where the goal is to eliminate the need for optimizing the latent
space of VQGAN for each input prompt.

- The training code is heavily based on the VQGAN-CLIP notebook <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>, thanks
to all the authors who contributed to the notebook (@crowsonkb, @advadnoun, @Eleiber, @Crimeacs, @Abulafia)
- Thanks to @lucidrains, the MLP mixer model (`mlp_mixer_pytorch.py`)  is from <https://github.com/lucidrains/mlp-mixer-pytorch>.
- Thanks to Taming Transformers authors <https://github.com/CompVis/taming-transformers>, the code uses VQGAN pre-trained model and
VGG16 feature space perceptual loss <https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py>
- Thanks to @afiaka87 for all the contributions to the repository's code and for providing the blog captions dataset for experimentation
- Thanks to VitGAN authors, the VitGAN model is from <https://github.com/wilile26811249/ViTGAN>
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
import json
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

import clip
from clip import simple_tokenizer
from mlp_mixer_pytorch import Mixer
from vitgan import Generator as VitGAN

from omegaconf import OmegaConf

try:
    import horovod.torch as hvd
    USE_HOROVOD = True
except ImportError:
    USE_HOROVOD = False

decode = simple_tokenizer.SimpleTokenizer().decode
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_DIM = 512
CLIP_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

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
    def __init__(self, cut_size, cutn, cut_pow=1., augs=None, pool=True):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.pool = pool

        # Parametrization of the augmentations and new augmentations taken from <https://github.com/nerdyrodent/VQGAN-CLIP>, thanks to @nerdyrodent.
        if not augs:
            augs = ('Af', 'Pe', 'Ji', 'Er')
        # self.augs = nn.Sequential(
            # # K.RandomHorizontalFlip(p=0.5),
            # # K.RandomVerticalFlip(p=0.5),
            # # K.RandomSolarize(0.01, 0.01, p=0.7),
            # # K.RandomSharpness(0.3,p=0.4),
            # # K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5),
            # # K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5),
            # K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            # K.RandomPerspective(0.7,p=0.7),
            # K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            # K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
        # )
        augment_list = []
        for item in augs:
            if item == 'Ji2':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05, p=0.5))
            elif item == 'Ji':
                augment_list.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.4, p=0.7))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'))
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.3, same_on_batch=False, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=1.0))
        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        if self.pool:
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            batch = cutout.repeat(self.cutn, 1, 1, 1)
        else:
            batch = input.repeat(self.cutn, 1, 1, 1)
        # for _ in range(self.cutn):
            # cutout = (self.av_pool(input) + self.max_pool(input))/2
            # cutouts.append(cutout)
        # batch = torch.cat(cutouts, dim=0)
        batch = self.augs(batch)
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


def tokenize(paths, out="tokenized.pkl", max_length:int=None):
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
    T = clip.tokenize(texts, truncate=True)
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
    toks = load_dataset(config.path) 
    print(f"Number of text prompts:{len(toks)}")

    vqgan_config = config.vqgan_config
    vqgan_checkpoint = config.vqgan_checkpoint
    clip_model = config.clip_model 
    lr = config.lr
    epochs = config.epochs

    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    clip_dim = CLIP_DIM
    clip_size = CLIP_SIZE
    vq_channels = model.quantize.embedding.weight.shape[1]
    vq_image_size = config.get("vq_image_size", 16) # if bigger, resolution of generated image is bigger
    noise_dim = config.noise_dim
    
    model_path = os.path.join(config.folder, "model.th")
    
    if os.path.exists(model_path):
        print(f"Resuming from {model_path}")
        net = torch.load(model_path, map_location="cpu")
    else:
        if config.model_type == "vitgan":
            net = VitGAN(
                initialize_size = vq_image_size//8, 
                dropout = config.dropout, 
                out_channels=vq_channels,
                input_dim=clip_dim+noise_dim,
                dim=config.dim,
                num_heads=config.get("num_heads", 6),
                blocks=config.depth,
            )
        elif config.model_type == "mlp_mixer":
            net = Mixer(
                input_dim=clip_dim+noise_dim, 
                image_size=vq_image_size, 
                channels=vq_channels, 
                patch_size=1, 
                dim=config.dim, 
                depth=config.depth, 
                dropout=config.dropout
            )
        else:
            raise ValueError("model_type should be 'vitgan' or  'mlp_mixer'")
    net = net.to(device)
    net.config = config
    opt = optim.Adam(net.parameters(), lr=lr)
    log_interval = config.get("log_interval", 100)

    rank_zero =  (USE_HOROVOD and hvd.rank() == 0) or not USE_HOROVOD
    if rank_zero:
        log_writer = SummaryWriter(config.folder)
    else:
        log_writer = None

    if USE_HOROVOD:
        opt = hvd.DistributedOptimizer(opt)
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(opt, root_rank=0)

    mean = torch.Tensor(CLIP_MEAN).view(1,-1,1,1).to(device)
    std = torch.Tensor(CLIP_STD).view(1,-1,1,1).to(device)
    cutn = config.cutn
    make_cutouts = MakeCutouts(clip_size, cutn=cutn, augs=config.get("augs"), pool=config.get("pool", True))
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    bs = config.batch_size
    repeat = config.repeat
    nb_noise = config.nb_noise
    dataset = torch.utils.data.TensorDataset(toks)
    diversity_mode = config.get("diversity_mode", "between_same_prompts")

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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=0, sampler=sampler, shuffle=shuffle)
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
    flow = torch.load("../clip_text_to_image_features/model.th", map_location="cpu").to(device)
    for epoch in range(epochs):
        if USE_HOROVOD:
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
            H = flow.sample(H.view(repeat*bs,-1,1,1)).view(repeat*bs,-1)
            if noise_dim:
                if nb_noise:
                    inds = np.arange(len(NOISE))
                    np.random.shuffle(inds)
                    inds = inds[:repeat]
                    noise = NOISE[inds].to(device).repeat(bs, 1).view(bs,repeat,-1).permute(1,0,2).contiguous().view(bs*repeat,-1)
                else:
                    noise = torch.randn(len(H), noise_dim).to(device)
                Hi = torch.cat((H,noise),dim=1)
            else:
                Hi = H
            z = net(Hi)
            #bs, vq_channels, vq_image_size, vq_image_size
            z = z.contiguous()
            z = z.view(repeat*bs, vq_channels, vq_image_size, vq_image_size)
            # print(z.min(),z.max())
            z = clamp_with_grad(z, z_min.min(), z_max.max())
            #repeat*bs, 3, h, w
            xr = synth(model, z)
            if config.diversity_coef:
                div = 0
                # for feats in [z]:
                for feats in lpips.net( (xr-mean)/std):
                    if diversity_mode == "between_same_prompts":
                        feats = normalize_tensor(feats)
                        _, cc,hh,ww = feats.shape
                        div += ( (feats.view(repeat, 1, bs, cc,hh,ww) - feats.view(1, repeat, bs, cc,hh,ww)) ** 2).sum(dim=3).mean()
                    elif diversity_mode == "all":
                        feats = normalize_tensor(feats)
                        _, cc,hh,ww = feats.shape
                        nb = len(feats)
                        div += ( (feats.view(nb, 1, cc,hh,ww) - feats.view(1, nb, cc,hh,ww)) ** 2).sum(dim=2).mean()
                    else:
                        raise ValueError("diversity_mode should be 'between_same_prompts' lr 'all'")
            else:
                div = torch.Tensor([0.]).to(device)
            #cutn*repeat*bs,3,h,w
            x = make_cutouts(xr)
            x = (x-mean)/std
            #cutn*repeat*bs,clip_dim
            embed = perceptor.encode_image(x).float() # generated image features
            #cutn*repeat*bs,clip_dim
            H = H.repeat(cutn, 1)
            H = H.view(cutn, repeat, bs, clip_dim)
            H = F.normalize(H, dim=-1)
            #cutn*repeat*bs,clip_dim
            H = H.view(-1, clip_dim)
            
            #cutn*repeat*bs,clip_dim
            embed = F.normalize(embed, dim=1)
            
            #dist between prompt features `H` and generated image features `embed`
            dists = (H.sub(embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean()
            opt.zero_grad()

            # 1) minimize distance between generated images CLIP features and text prompt features
            # 2) maximize diversity of the generated images
            loss = dists  - config.diversity_coef * div
            loss.backward()
            opt.step()
            if rank_zero: 
                log_writer.add_scalar("loss", loss.item(), step)
                log_writer.add_scalar("dists", dists.item(), step)
                log_writer.add_scalar("diversity", div.item(), step)
            avg_loss = loss.item() * 0.01 + avg_loss * 0.99 
            if rank_zero and step % log_interval == 0:
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


def test(model_path, text_or_path, *, nb_repeats=1, out_path="gen.png", images_per_row:int=None):
    """
    generated an image or a set of images from a model given a text prompt

    model_path: str
        path of the model
    
    text_or_path: str
        can either be:
         - a text prompt. several text prompts can be provided  by delimiting them using "|"
         - a path to a text file .txt, where each line is a text prompt
    
    nb_repeats: int
        number of times the same text prompt is repeated
        with different noise vectors
    
    out_path: str
        output path
    
    images_per_row: int
        number of images per row in the grid of images
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.load(model_path, map_location="cpu").to(device)
    config = net.config
    vqgan_config = config.vqgan_config 
    vqgan_checkpoint = config.vqgan_checkpoint
    clip_model = config.clip_model
    clip_dim = CLIP_DIM
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    
    if text_or_path.endswith(".txt"):
        texts = [t.strip() for t in open(text_or_path).readlines()]
    else:
        texts = text_or_path.split("|")
    toks = clip.tokenize(texts, truncate=True)
    H = perceptor.encode_text(toks.to(device)).float()
    H = H.repeat(nb_repeats, 1)
    noise_dim = net.config.noise_dim
    if noise_dim:
        if hasattr(net, "NOISE"):
            noise = net.NOISE
            if len(noise) > len(H):
                noise = noise[:len(H)]
            else:
                inds = np.random.randint(0, len(noise), size=len(H))
                noise = noise[inds]
            noise = noise.to(device)
        else:
            noise = torch.randn(len(H), noise_dim).to(device)
        H = torch.cat((H, noise), dim=1)
    with torch.no_grad():
        z = net(H)
        z = clamp_with_grad(z, z_min.min(), z_max.max())
        xr = synth(model, z)
    grid = torchvision.utils.make_grid(xr.cpu(), nrow=images_per_row if images_per_row else nb_repeats)
    TF.to_pil_image(grid).save(out_path)

@torch.no_grad()
def evaluate(model_path, data_path, *, batch_size:int=None, out_folder=None, clip_threshold=40, nb_test:int=None, save_images=False, img_folder=None, images_per_row=8, seed=42):
    """
    Evaluate the CLIP score of a model on a dataset of prompts.
    It also optionally saves the generated images of the prompts.

    model_path: str
        path to model

    data_path: dataset path
        like in train, could be a pkl or a text file or a glob pattern of text files

    batch_size: int
        mini-batch used for evaluation

    out_folder: str
        folder where to save the results, default is
        <model_dir>.
        what is saved is:
            - <model_dir>/eval_<dataset_name>.th which contains all the CLIP score of each prompt
            - <model_dir>/eval_<dataset_name>.json which contains average CLIP score

    clip_threshold: int
        threshold for CLIP score used for evaluation

    nb_test: int
        Number of examples to use for evaluation, can be used
        to filter the dataset if it is too big
    
    save_images: bool
        whether to save the generated images

    img_folder: str
        path where to save the images if save_images is True, default is
        <model_dir>/eval_<dataset_name>_images

    images_per_row: int
        number of images per row in the grid of images if save_images is True

    seed: int
        seed used to subsample the prompt dataset
    
    """
    name = os.path.basename(path)
    if not out_folder:
        out_folder = os.path.dirname(model_path)
        os.makedirs(out_folder, exist_ok=True)
    if not img_folder:
        img_folder = os.path.join(os.path.dirname(model_path), f"eval_{name}_images")
        os.makedirs(img_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.load(model_path, map_location="cpu").to(device)
    config = net.config
    vqgan_config = config.vqgan_config 
    vqgan_checkpoint = config.vqgan_checkpoint
    clip_model = config.clip_model
    clip_dim = CLIP_DIM
    clip_size = CLIP_SIZE
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    mean = torch.Tensor(CLIP_MEAN).view(1,-1,1,1).to(device)
    std = torch.Tensor(CLIP_STD).view(1,-1,1,1).to(device)
    
    toks = load_dataset(data_path)
    if not batch_size:
        batch_size = config.batch_size
        
    np.random.seed(seed)
    if nb_test:
        inds = np.arange(len(toks))
        np.random.shuffle(inds)
        inds = inds[:nb_test]
        toks = toks[inds]
    print(f"Evaluate on {len(toks)} prompts...")
    dataset = torch.utils.data.TensorDataset(toks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    clip_scores_batches = []
    noise_dim = net.config.noise_dim
    logits_scale = perceptor.logit_scale.exp().to(device)
    for batch_idx, (tok,) in enumerate(dataloader):
        tok = tok.to(device)
        H = perceptor.encode_text(tok).float()
        if noise_dim:
            if hasattr(net, "NOISE"):
                noise = net.NOISE
                if len(noise) > len(H):
                    noise = noise[:len(H)]
                else:
                    inds = np.random.randint(0, len(noise), size=len(H))
                    noise = noise[inds]
                noise = noise.to(device)
            else:
                noise = torch.randn(len(H), noise_dim).to(device)
            Hi = torch.cat((H, noise), dim=1)
        else:
            Hi = H
        z = net(Hi)
        z = clamp_with_grad(z, z_min.min(), z_max.max())
        xr = synth(model, z)
        if save_images:
            grid = torchvision.utils.make_grid(xr.cpu(), nrow=images_per_row)
            TF.to_pil_image(grid).save(os.path.join(img_folder, f"batch_{batch_idx:010d}.png"))
            text = "\n".join([decode(t.tolist()) for t in tok])
            with open(os.path.join(img_folder, f"batch_{batch_idx:010d}.txt"), "w") as fd:
                fd.write(text)
        xr = torch.nn.functional.interpolate(xr, size=(clip_size, clip_size), mode='bilinear')
        xr = (xr - mean) / std
        embed = perceptor.encode_image(xr).float()
        image_features = F.normalize(embed, dim=1)
        text_features = F.normalize(H, dim=1) 
        clip_scores = (logits_scale * (image_features * text_features).sum(dim=1)).cpu()
        clip_scores_batches.append(clip_scores)
    clip_scores = torch.cat(clip_scores_batches)


    out = os.path.join(out_folder, f"eval_{name}.th")
    print(f"Saving to {out}")
    torch.save(clip_scores, out)

    mean = clip_scores.mean().item()
    std = clip_scores.std().item()
    clip_score_atleast = (clip_scores >= clip_threshold).float().mean().item()
    
    dump = {
        "clip_score_mean": mean,
        "clip_score_std": std,
        f"clip_score_atleast_{clip_threshold}": clip_score_atleast,
    }
    out = os.path.join(out_folder, f"eval_{name}.json")
    print(f"Saving to {out}")
    with open(out, "w") as fd:
        fd.write(json.dumps(dump))
    print(f"CLIP score mean: {mean}. CLIP score std:{std}")
    print(f"Fraction of images with a CLIP score of at least {clip_threshold}: {clip_score_atleast}")

def load_dataset(path):
    # path can be the following:
    # - a path to a text file where each line is a text prompt
    # - a glob pattern (*) of text files where each file is a text prompt
    # - a pkl file created using `tokenize`, where each text prompt is already tokenized
    if path.endswith("pkl"):
        toks = torch.load(path)
    elif "*" in path:
        texts = [open(f).read().strip() for f in glob(path)]
        toks = clip.tokenize(texts, truncate=True)
    else:
        texts = [t.strip() for t in open(path).readlines()]
        toks = clip.tokenize(texts, truncate=True)
    return toks

if __name__ == "__main__":
    run([train, test, tokenize, encode_images, evaluate])
