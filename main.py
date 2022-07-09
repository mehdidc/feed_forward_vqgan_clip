"""
Feed forward VQGAN-CLIP model, where the goal is to eliminate the need for optimizing the latent
space of VQGAN for each input prompt.
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
import kornia

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from taming.models import cond_transformer, vqgan
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.lpips import normalize_tensor

import clip
from clip import simple_tokenizer
from mlp_mixer_pytorch import Mixer
from vitgan import Generator as VitGAN, SimpleGenerator as SimpleVitGAN
from transformer import XTransformer
from cloob import CLOOB
from omegaconf import OmegaConf

if os.getenv("USE_HOROVOD") == "false":
    USE_HOROVOD = False
else:
    try:
        import horovod.torch as hvd
        USE_HOROVOD = True
    except ImportError:
        USE_HOROVOD = False

decode = simple_tokenizer.SimpleTokenizer().decode
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_SIZE = {
    "RN50": 224,
    "RN101": 224,
    "RN50x4": 288,
    "RN50x16": 384,
    "ViT-B/32": 224,
    "ViT-B/16": 224,
    "ViT-L/14": 224,
    "cloob_rn50": 224,
    "cloob_rn50x4": 288,
    "cloob_laion_400m_vit_b_16_32_epochs": 224,
    "openclip/ViT-B-32-quickgelu/laion400m_e32": 224,
    "openclip/ViT-B-32/laion2b_e16": 224,
}
CLIP_DIM = {
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "RN50x16": 768,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "cloob_rn50": 1024,
    "cloob_rn50x4": 640,
    "cloob_laion_400m_vit_b_16_32_epochs": 512,
    "openclip/ViT-B-32-quickgelu/laion400m_e32": 512,
    "openclip/ViT-B-32/laion2b_e16": 512,
}
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
        model.quantize.embedding = model.quantize.embed
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
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    x  = clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)
    return x

class Resize(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, (self.size, self.size), mode="bilinear")

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., pool_size=None, interp_size=None, augs=None, pool=True, interpolate=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.pool = pool
        self.interpolate = interpolate
        self.pool_size = pool_size
        # Parametrization of the augmentations and new augmentations taken from <https://github.com/nerdyrodent/VQGAN-CLIP>, thanks to @nerdyrodent.
        if not augs:
            augs = ('Af', 'Pe', 'Ji', 'Er')

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
            elif item == 'Er2':
                augment_list.append(K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=False, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=1.0))
            elif item == 'Re2':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.9,1),  ratio=(0.75,1.333), cropping_mode='resample', p=1.0))
            elif item == 'Cc':
                augment_list.append(K.CenterCrop(size=(self.cut_size,self.cut_size), p=1.0))
            elif item == 'R':
                augment_list.append(Resize(self.cut_size))
        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        if pool_size is None:
            pool_size = cut_size
        if interp_size is None:
            interp_size = pool_size
        self.pool_size = pool_size
        self.interp_size = interp_size
        self.av_pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))

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
        batch = self.augs(batch)
        if self.noise_fac:
            facs = batch.new_empty([len(batch), 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        if self.interpolate:
            # batch = torch.nn.functional.interpolate(batch, size=(self.interp_size, self.interp_size), mode="bicubic")
            batch = torch.nn.functional.adaptive_avg_pool2d(batch, (self.interp_size, self.interp_size))
        return batch

def encode_text_and_images(
    folder, *, img_ext="jpg", text_ext="txt", out="features.pkl", 
    clip_model="ViT-B/32", 
    clip_path:str=None
):
    """
    encode (text,image) pairs to CLIP features
    can be used to train a text to image model.

    folder: str
        folder with text and images.
        consist in a set of pairs of files, e.g.,
            - file1.jpg file1.txt
            - file2.jpg file2.jpg
            - ...
    img_ext: str
        image extension
    text_text: str
        text extension

    out: str
        output pkl file
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model = load_clip_model(clip_model, path=clip_path)

    text_paths = glob(os.path.join(folder, "*."+text_ext))
    img_paths = [t.replace(text_ext, img_ext) for t in text_paths]
    
    text_features_list = []
    image_features_list = []

    for text_path, img_path in zip(text_paths, img_paths):
        text = open(text_path).read()
        text_toks = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_toks)
        text_features_list.append(text_features.cpu())

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features_list.append(image_features.cpu())

    text_features = torch.cat(text_features_list)
    image_features = torch.cat(image_features_list)
    torch.save((text_features, image_features), out)


def encode_text_and_images_webdataset(
    pattern, *, 
    clip_model="ViT-B/32", clip_path:str=None, 
    batch_size=512, 
    img_col="input.jpg", txt_col="output.txt", 
    out="features.pkl",
    num_workers=8,
    image_quality_threshold:float=None,
    image_quality_method='nima',
    merge=False,
):
    """
    encode (text,image) pairs to CLIP features from webdataset.
    can be used to train a text to image model.

    Can optionally filter images according to a quality proxy metric
    from `pyiqa`, if available. Check https://github.com/chaofengc/IQA-PyTorch 
    for more info.
    """
    import webdataset as wds
    from PIL import Image
    from io import BytesIO
    if USE_HOROVOD:
        hvd.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and USE_HOROVOD:
        torch.cuda.set_device(hvd.local_rank())
    try:
        from pyiqa.models.inference_model import InferenceModel
        iqa_model = InferenceModel(image_quality_method, '')
    except Exception:
        pass

    _, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model = load_clip_model(clip_model, path=clip_path).eval().to(device)
    def transform_image(x):
        return preprocess(x)
    def transform_text(x):
        return x
    def filter_dataset(item):
        try:
            x = Image.open(BytesIO(item[img_col]))
        except Exception as ex:
            print(ex)
            return False
        else:
            return True
    tars = glob(pattern)
    tars = sorted(tars)
    if USE_HOROVOD:
        tars = [t for i,t in enumerate(tars) if i % hvd.size() == hvd.rank()]
    ds = wds.WebDataset(tars, handler=wds.warn_and_continue)
    ds = ds.select(filter_dataset)
    ds = ds.decode("pil")
    ds = ds.to_tuple(img_col, txt_col)
    ds = ds.map_tuple(transform_image, transform_text)
    ds = ds.batched(batch_size)
    dl = wds.WebLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers)
    mean = torch.Tensor(CLIP_MEAN).view(1,-1,1,1).to(device)
    std = torch.Tensor(CLIP_STD).view(1,-1,1,1).to(device)
    imf = []
    txf = []
    nb = 0
    for X, Y in dl:
        Y = clip.tokenize(Y, truncate=True)
        X = X.to(device)
        Y = Y.to(device)
        if image_quality_threshold is not None:
            scores = iqa_model.net(X*std+mean).flatten()
            good_quality = scores>=image_quality_threshold
            X = X[good_quality]
            Y = Y[good_quality]
            if len(X) == 0:
                continue
        #if hvd.rank() == 0:
        #    print(X.shape)
        with torch.no_grad():
            image_features = model.encode_image(X).cpu()
            text_features = model.encode_text(Y).cpu()
        imf.append(image_features)
        txf.append(text_features)
        nb += len(X)
        if hvd.rank() == 0:
            print(nb)
    if USE_HOROVOD:
        nb = torch.Tensor([nb]).long()
        nb = hvd.allreduce(nb, average=False)
        nb = nb.item()
    print("Nb of images processed:", nb)
    imf = torch.cat(imf)
    txf = torch.cat(txf)
    if USE_HOROVOD:
        idx = hvd.rank()
        torch.save((txf,imf), f"{out}_{idx}")
        hvd.join()
        if merge:
            if hvd.rank() == 0:
                xs = []
                ys = []
                paths = [f"{out}_{idx}" for idx in range(hvd.size())]
                for path in paths:
                    x, y = torch.load(path)
                    xs.append(x)
                    ys.append(y)
                xs = torch.cat(xs)
                ys = torch.cat(ys)
                torch.save((xs,ys), out)
                for path in paths:
                    os.remove(path)
            hvd.join()
    else:
        torch.save((txf,imf), out)


def tokenize(paths, out="tokenized.pkl", max_length:int=None, batch_size=None):
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
    if batch_size is None:
        batch_size = len(texts)
    toks = []
    for i in range(0, len(texts), batch_size):
        T = clip.tokenize(texts[i:i+batch_size], truncate=True)
        toks.append(T)
    toks = torch.cat(toks)
    torch.save(toks, out)

def tv_loss(Y_hat):
    """
    Total variation loss
    """
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


def _fix_mlp_mixer_gelu_issue(net):
    # Solving https://github.com/mehdidc/feed_forward_vqgan_clip/issues/25 for torch>=1.12
    # Thanks to @neverix for the solution
    for l in net.mixer:
        if isinstance(l, torch.nn.Sequential):
            for k in l:
                k.fn[1].approximate = "none"
    return net

def _fix_vitgan_gelu_issue(net):
    # Solving https://github.com/mehdidc/feed_forward_vqgan_clip/issues/25 for torch>=1.12
    # Thanks to @neverix for the solution
    for block in net.Transformer_Encoder.blocks:
            block.mlp.activation.approximate = "none"
    return net


def build_model(config):
    clip_model = config.clip_model 
    clip_size = config.get("clip_size", CLIP_SIZE.get(clip_model))
    clip_dim = config.get("clip_dim", CLIP_DIM.get(clip_model))
    vq = load_vqgan_model(config.vqgan_config, config.vqgan_checkpoint)

    vq_config = OmegaConf.load(config.vqgan_config)
    vq_channels = vq_config.model.params.ddconfig.z_channels
    vq_image_size = config.get("vq_image_size", 16) # if bigger, resolution of generated image is bigger
    noise_dim = config.noise_dim

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
    elif config.model_type == "simple_vitgan":
        net = SimpleVitGAN(
            size=vq_image_size,
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
    elif config.model_type == "xtransformer":
        net = XTransformer(
            input_dim=clip_dim+noise_dim, 
            image_size=vq_image_size, 
            channels=vq_channels, 
            dim=config.dim, 
            depth=config.depth, 
            heads=config.get("num_heads", 6),
            initial_proj=config.get("initial_proj", True),
            add_input=config.get("add_input", False)
        )
    else:
        raise ValueError("model_type should be 'vitgan' or  'mlp_mixer' or 'xtransformer'")
    return net

def train(config_file):

    config = OmegaConf.load(config_file)
    if not hasattr(config, "folder"):
        config.folder = os.path.dirname(config_file)
    use_wandb = config.get("use_wandb", False)
    use_ema = config.get("use_ema", False)
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.get("wandb_project", "feed_forward_vqgan_clip"),
            entity=config.get("wandb_entity"),
            resume=False,
            config=config,
        )
        wandb_log_interval = config.get("wandb_log_interval", 1)
    if use_ema:
        """
        EMA improves the eval metrics a little bit
        """
        from torch_ema import ExponentialMovingAverage
        ema_decay = config.get("ema_decay", 0.995)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if USE_HOROVOD:
        hvd.init()
        if device == "cuda":
            torch.cuda.set_device(hvd.local_rank())
    if config.diversity_coef:
        # VGG can be used for maximizing diversity
        # on feature space
        lpips = LPIPS()
        lpips.load_from_pretrained()
        lpips = lpips.to(device)

    # Load dataset
    toks = load_dataset(config.path) 
    vqgan_config = config.vqgan_config
    vqgan_checkpoint = config.vqgan_checkpoint
    clip_model = config.clip_model 
    lr = config.lr
    epochs = config.epochs

    # Load VQGAN
    vq = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)

    # Load CLIP
    perceptor = load_clip_model(clip_model, path=config.get("clip_model_path"))
    perceptor = perceptor.to(device)
    clip_size = config.get("clip_size", CLIP_SIZE.get(clip_model))
    clip_dim = config.get("clip_dim", CLIP_DIM.get(clip_model))
    vq_channels = vq.quantize.embedding.weight.shape[1]

    vq_image_size = config.get("vq_image_size", 16) # if bigger, resolution of generated image is bigger
    noise_dim = config.noise_dim
    
    # Previously, the model instance was directly saved into `model.th`, keep support
    # these for backward compatibility. 
    # From now on, we rather save the state dict directly into `checkpoint.th`, with config, epoch and step
    # information.
    model_path = os.path.join(config.folder, "model.th")
    checkpoint_path = os.path.join(config.folder, "checkpoint.th")
    
    # Build Model that will map text embedding to VQGAN latent space
    if os.path.exists(model_path):
        # backward compability
        print(f"Resuming from {model_path}")
        net = torch.load(model_path, map_location="cpu")
        if net.config.model_type == "mlp_mixer":
            _fix_mlp_mixer_gelu_issue(net)
        elif net.config.model_type == "vitgan":
            _fix_vitgan_mixer_gelu_issue(net)
    else:
        net = build_model(config)
        if os.path.exists(checkpoint_path):
            # We load the state dict (current way, instead of saving/loading model instance)
            print(f"Resuming model from checkpoint {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            net.load_state_dict(ckpt["state_dict"])
            net.epoch = ckpt['epoch']
            net.step = ckpt['step']
    net = net.to(device)
    if not hasattr(net, "step"):
        net.step = 0
    if not hasattr(net, "epoch"):
        net.epoch = 0
    net.config = config
    opt = optim.Adam(net.parameters(), lr=lr)
    opt_path = os.path.join(config.folder, "opt.th")
    # Load optimizer state
    if os.path.exists(opt_path):
        print(f"Resuming optimizer state from {opt_path}")
        opt.load_state_dict(torch.load(opt_path, map_location="cpu"))

    # Load EMA (expoential moving average) parameters
    if use_ema:
        # Support loading model instance for backward compability.
        # From now on, we only state the state dict into `checkpoint_ema.th`.
        model_ema_path = os.path.join(config.folder, "model_ema.th")
        checkpoint_ema_path = os.path.join(config.folder, "checkpoint_ema.th")
        if os.path.exists(model_ema_path):
            # backward compability
            model_ema = torch.load(model_ema_path, map_location="cpu").to(device)
            ema = ExponentialMovingAverage(model_ema.parameters(), decay=ema_decay)
        elif os.path.exists(checkpoint_ema_path):
            # current way, use state dicts
            ckpt = torch.load(checkpoint_ema_path, map_location='cpu')
            net_ema = build_model(config)
            net_ema.load_state_dict(ckpt['state_dict'])
            ema = ExponentialMovingAverage(net.parameters(), decay=ema_decay) 
        else:
            ema = ExponentialMovingAverage(net.parameters(), decay=ema_decay)
        ema.to(device)

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
    cut_size = config.get("cut_size", clip_size)

    # Data augmentation
    make_cutouts = MakeCutouts(
        cut_size=cut_size, cutn=cutn, 
        augs=config.get("augs"), 
        pool=config.get("pool", True), 
        pool_size=config.get("pool_size", clip_size),
        interpolate=config.get("interpolate", False), 
        interp_size=config.get("interp_size", clip_size),
    )
    z_min = vq.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = vq.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    bs = config.batch_size
    repeat = config.repeat
    nb_noise = config.nb_noise
    
    # Load dataset
    if type(toks) == tuple:
        dataset = torch.utils.data.TensorDataset(*toks)
    else:
        dataset = torch.utils.data.TensorDataset(toks, toks)
    print(f"Number of examples:{len(dataset)}")

    diversity_mode = config.get("diversity_mode", "between_same_prompts")

    # Fast evaluation based on CLIP generate image/text similarity on some prompts
    if config.get("eval_path"):
        eval_data = load_dataset(config.eval_path) 
        eval_perceptor = load_clip_model(config.eval_clip_model).to(device) if config.get("eval_clip_model") else perceptor
    else:
        eval_data = None
        eval_perceptor = None

    if USE_HOROVOD:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=hvd.size(),
            rank=hvd.rank(),
        )
        shuffle = None
    else:
        sampler = None
        shuffle = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=0, sampler=sampler, shuffle=shuffle)
    first_batch = next(iter(dataloader))
    if nb_noise:
        if hasattr(net, "NOISE"):
            NOISE = net.NOISE
        else:
            NOISE = torch.randn(nb_noise,noise_dim)
        if USE_HOROVOD:
            NOISE = hvd.broadcast(NOISE, root_rank=0)
        net.NOISE = NOISE
    
    # Training hyper-parameters
    input_loss = config.get("input_loss", False)
    input_loss_coef = config.get("input_loss_coef", 1)
    target_loss_coef = config.get("target_loss_coef", 1)
    clip_grad_norm = config.get("clip_grad_norm")
    avg_loss = 1. 
    step = net.step
    normalize_input = config.get("normalize_input", False)
    l2_coef = config.get("l2_coef", 0.)
    tv_coef = config.get("tv_coef", 0.)
    tv_exponent = config.get("tv_exponent", 1.)
    logits_scale = eval_perceptor.logit_scale.exp().cpu() if eval_perceptor is not None else None
    # Load scheduler
    if config.get("scheduler") is not None:
        if config.scheduler == "cosine":
            steps = config.max_steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0, last_epoch=-1, verbose=False)
        else:
            raise ValueError(config.scheduler)
    else:
        scheduler = None

    # Start training
    for epoch in range(net.epoch, epochs):
        if USE_HOROVOD:
            sampler.set_epoch(epoch)
        for inp, out in dataloader:
            # `inp``: text embedding or text tokens or image embedding
            # `out`: text embedding or text tokens or image embedding
            # For most cases, `inp` and `out` are just the same, e.g.,
            # when the dataset is just a list of prompts, `inp` and `out`
            # are the same. But it is also possible to construct a dataset
            # where `inp` is the text embeddings and `out` is the image embedding
            # computed from a dataset of image-text pairs.
            # The model model takes `inp` as input, and generates an image.
            # We then compute image embeddings from the generated image.
            # Then, we minimize the distance between the image embeddings
            # of the generated image and the embeddings of `inp` (text embeddings from dataset).
            # Additionally (optional), we can also minimize the distance
            # between the generated image embeddings and `out` (image embeddings from dataset).
            inp = inp.to(device)
            out = out.to(device)
            bs = len(inp)
            #bs,clip_dim
            inp_feats = perceptor.encode_text(inp).float() if inp.dtype == torch.long else inp.float()
            if normalize_input:
                inp_feats = F.normalize(inp_feats, dim=1)
            #bs,clip_dim
            out_feats = perceptor.encode_text(out).float() if out.dtype == torch.long else out.float()
            #repeat*bs,clip_dim
            inp_feats = inp_feats.repeat(repeat, 1)
            out_feats = out_feats.repeat(repeat, 1)
            if noise_dim:
                if nb_noise:
                    inds = np.arange(len(NOISE))
                    np.random.shuffle(inds)
                    inds = inds[:repeat]
                    noise = NOISE[inds].to(device).repeat(bs, 1).view(bs,repeat,-1).permute(1,0,2).contiguous().view(bs*repeat,-1)
                else:
                    noise = torch.randn(len(inp_feats), noise_dim).to(device)
                inp_feats_net = torch.cat((inp_feats,noise),dim=1)
            else:
                inp_feats_net = inp_feats
            
            # Use the model to predict the vqgan latent space `z`
            z = net(inp_feats_net)
            #bs, vq_channels, vq_image_size, vq_image_size
            z = z.contiguous()
            z = z.view(repeat*bs, vq_channels, vq_image_size, vq_image_size)
            if l2_coef > 0:
                # L2 loss (optional)
                l2 = (z**2).mean()
            else:
                l2 = torch.Tensor([0.]).to(device)
            z = clamp_with_grad(z, z_min.min(), z_max.max())

            # Generate the image from the VQGAN latent space
            #repeat*bs, 3, h, w
            xr = synth(vq, z)

            if tv_coef > 0:
                # Total variation loss (optional)
                tv = tv_loss(xr)
            else:
                tv = torch.Tensor([0.]).to(device)

            # Diversity loss
            if config.diversity_coef:
                div = 0
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

            # Generate random augmentations from the generated images

            #cutn*repeat*bs,3,h,w
            x = make_cutouts(xr)
            x = (x-mean)/std
            #cutn*repeat*bs,clip_dim
            embed = perceptor.encode_image(x).float() # generated image features
            #cutn*repeat*bs,clip_dim
            H = out_feats.repeat(cutn, 1)
            H = H.view(cutn, repeat, bs, clip_dim)
            H = F.normalize(H, dim=-1)
            #cutn*repeat*bs,clip_dim
            H = H.view(-1, clip_dim)
            
            #cutn*repeat*bs,clip_dim
            embed = F.normalize(embed, dim=1)
            
            #dist between prompt features `H` and generated image features `embed`
            dists = target_loss_coef * ((H.sub(embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean())
            if input_loss:
                # If dataset provided is pairs of embeddings, we have source and target embeddings, which can
                # typically be (text , image) pair embeddings.
                # By default, in this case, we minimize the distance between generated image embeddings and target embeddings 
                # (i.e, image embeddings) in the dataset. 
                # We can also optionally minimize distance between generated image embeddings and source embeddings (i.e, text embeddings)
                # in the dataset by making `input_loss_coef` > 0
                H = inp_feats.repeat(cutn, 1)
                H = H.view(cutn, repeat, bs, clip_dim)
                H = F.normalize(H, dim=-1)
                #cutn*repeat*bs,clip_dim
                H = H.view(-1, clip_dim)
                dists += input_loss_coef * ((H.sub(embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean())
            opt.zero_grad()

            # 1) minimize distance between generated images CLIP features and text/image prompt features
            # 2) maximize diversity of the generated images
            # 3) L2 loss (optional)
            # 4) Total Variation loss (optional)
            loss = dists  - config.diversity_coef * div  + l2_coef * l2 + tv_coef * tv
            loss.backward()
            if clip_grad_norm:
                clip_grad_norm_(net.parameters(), clip_grad_norm)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if USE_HOROVOD:
                loss = hvd.allreduce(loss)
                dists = hvd.allreduce(dists)
                div = hvd.allreduce(div)
                l2 = hvd.allreduce(l2)
            if rank_zero and use_ema:
                ema.update()
            if rank_zero: 
                log_writer.add_scalar("loss", loss.item(), step)
                log_writer.add_scalar("dists", dists.item(), step)
                log_writer.add_scalar("diversity", div.item(), step)
                log_writer.add_scalar("l2", l2.item(), step)
                log_writer.add_scalar("tv", tv.item(), step)
                if use_wandb and step % wandb_log_interval == 0:
                    log = {
                        "avg_loss": avg_loss, 
                        "loss": loss.item(), 
                        "dists": dists.item(), 
                        "diversity": div.item(), 
                        "l2": l2.item(),
                        "tv": tv.item(),
                    }
                    wandb.log(log)
            avg_loss = loss.item() * 0.01 + avg_loss * 0.99 
            # Report progress

            if rank_zero and step % log_interval == 0:
                print(f"epoch:{epoch:03d}, step:{step:05d}, avg_loss:{avg_loss:.3f}, loss:{loss.item():.3f}, dists:{dists.item():.3f}, div:{div.item():.3f}, l2:{l2.item():.3f} tv:{tv.item()}")
                if eval_data is not None:
                    # Fast evaluation using CLIP text/image score/distance
                    bs = config.batch_size
                    eval_clip_score_list = []
                    eval_dists_list = []
                    for i in range(0, len(eval_data), bs):
                        text_emb = (
                            eval_perceptor.encode_text(eval_data[i:i+bs].to(device)).float() 
                            if eval_data.dtype == torch.long else eval_data[i:i+bs].float().to(device)
                        )
                        out_feats = text_emb
                        with torch.no_grad():
                            z = net(text_emb)
                            xr_eval = synth(vq, z)
                            xr_eval = torch.nn.functional.interpolate(xr_eval, size=(clip_size, clip_size), mode='bilinear')
                            xr_eval = (xr_eval - mean) / std
                            embed = eval_perceptor.encode_image(xr_eval).float() 
                            H = F.normalize(out_feats, dim=-1)
                            H = H.view(-1, clip_dim)
                            embed = F.normalize(embed, dim=1)
                            eval_dists = (H.sub(embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2))

                            eval_clip_score = (logits_scale * (H*embed).sum(dim=1))
                            eval_dists_list.append(eval_dists.cpu())
                            eval_clip_score_list.append(eval_clip_score.cpu())
                    eval_dists = torch.cat(eval_dists_list).mean()
                    eval_clip_score = torch.cat(eval_clip_score_list).mean()
                    print(f"Eval dists: {eval_dists:.3f}")
                    print(f"Eval clip score: {eval_clip_score:.3f}")
                    log_writer.add_scalar("eval_dists", eval_dists.item(), step)
                    log_writer.add_scalar("eval_clip_score", eval_clip_score.item(), step)
                else:
                    eval_dists = 0.0
                # Saves generated images of current batch
                grid = torchvision.utils.make_grid(xr.cpu(), nrow=bs)
                TF.to_pil_image(grid).save(os.path.join(config.folder, 'progress.png'))
                TF.to_pil_image(grid).save(os.path.join(config.folder, f'progress_{step:010d}.png'))
                net.step = step
                torch.save({"state_dict": net.state_dict(), "config": config, "step": step, "epoch": epoch}, checkpoint_path)
                if use_ema:
                    with ema.average_parameters():
                        torch.save(
                            {"state_dict": net.state_dict(), "config": config, "step": step, "epoch": epoch}, 
                            checkpoint_ema_path
                        )
                torch.save(opt.state_dict(), os.path.join(config.folder, "opt.th"))

                if inp.dtype == torch.long:
                    text = "\n".join([decode(t.tolist()) for t in inp])
                    with open(os.path.join(config.folder, "progress.txt"), "w") as fd:
                        fd.write(text)
                    with open(os.path.join(config.folder, f"progress_{step:010d}.txt"), "w") as fd:
                        fd.write(text)

                # Saves generated images of a fixed batchs
                inp_fixed_batch, out_fixed_batch = first_batch
                out_fixed_batch = out_fixed_batch.to(device)
                inp_fixed_batch = inp_fixed_batch.to(device)
                with torch.no_grad():
                    inp_feats = perceptor.encode_text(inp_fixed_batch).float() if inp_fixed_batch.dtype == torch.long else inp_fixed_batch.float()
                    out_feats = perceptor.encode_text(out_fixed_batch).float() if out_fixed_batch.dtype == torch.long else out_fixed_batch.float()

                    if normalize_input:
                        inp_feats = F.normalize(inp_feats, dim=1)
                    if noise_dim:
                        inp_feats = torch.cat((inp_feats, noise[:len(inp_feats)]), dim=1)
                    if use_ema:
                        with ema.average_parameters():
                            z = net(inp_feats)
                    else:
                        z = net(inp_feats)
                    #bs, vq_channels, vq_image_size, vq_image_size
                    z = z.contiguous()
                    z = z.view(len(z), vq_channels, vq_image_size, vq_image_size)
                    z = clamp_with_grad(z, z_min.min(), z_max.max())
                    #repeat*bs, 3, h, w
                    xr_fixed_batch = synth(vq, z)
                grid = torchvision.utils.make_grid(xr_fixed_batch.cpu(), nrow=bs)
                TF.to_pil_image(grid).save(os.path.join(config.folder, 'fixed_batch_progress.png'))
                TF.to_pil_image(grid).save(os.path.join(config.folder, f'fixed_batch_progress_{step:010d}.png'))
                if step == 0 and inp_fixed_batch.dtype == torch.long:
                    text = "\n".join([decode(t.tolist()) for t in inp_fixed_batch])
                    with open(os.path.join(config.folder, "fixed_batch.txt"), "w") as fd:
                        fd.write(text)

                if use_wandb:
                    # Report on wandb (alternative to tensorboard)
                    caption = [decode(t.tolist()) for t in inp] if inp.dtype == torch.long else None
                    caption_fixed_batch = [decode(t.tolist()) for t in inp_fixed_batch] if inp_fixed_batch.dtype == torch.long else None
                    xr = xr.view(repeat, bs, xr.shape[1], xr.shape[2], xr.shape[3]).cpu()
                    log = {}
                    log["image"] = [
                        wandb.Image(xr[r, i].cpu(), caption=caption[i] if caption else None)
                        for r in range(repeat)
                        for i in range(bs)
                    ]
                    log["image_fixed"] = [
                        wandb.Image(xr_fixed_batch[i].cpu(), caption=caption_fixed_batch[i] if caption_fixed_batch else None)
                        for i in range(len(xr_fixed_batch))
                    ]
                    wandb.log(log)
                    model_artifact = wandb.Artifact('trained-model', type='model', metadata=dict(net.config))
                    model_artifact.add_file(model_path)
                    wandb_run.log_artifact(model_artifact)
            step += 1
            # Posssbility to have fixed number of steps
            if config.get("max_steps") is not None and step >= config.max_steps:
                # finish
                return


def test(
    model_path, text_or_path, *, 
    nb_repeats=1, 
    out_path="gen.png", 
    images_per_row:int=None, 
    prior_path:str=None,
    seed:int=None,
):
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
    
    prior_path: str
        Path to flow, a model trained with `train_prior`, which generates image embeddings from text embeddings.
    
    seed: int
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(model_path)
    config = net.config
    net = net.to(device)    
    vqgan_config = config.vqgan_config 
    vqgan_checkpoint = config.vqgan_checkpoint
    clip_model = config.clip_model
    perceptor = load_clip_model(clip_model, path=config.get("clip_model_path"))
    perceptor = perceptor.to(device)
    vq = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    z_min = vq.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = vq.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    if prior_path:
        prior = load_prior_model(prior_path).to(device)
    if text_or_path.endswith(".txt"):
        texts = [t.strip() for t in open(text_or_path).readlines()]
    else:
        texts = text_or_path.split("|")
    normalize_input = config.get("normalize_input", False)
    toks = clip.tokenize(texts, truncate=True)
    H = perceptor.encode_text(toks.to(device)).float()
    if normalize_input:
        H = F.normalize(H, dim=1)
    H = H.repeat(nb_repeats, 1)
    if prior_path:
        H = H.view(len(H), -1, 1, 1)
        H = prior.sample(H)
        H = H.view(len(H), -1)
    noise_dim = config.noise_dim
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
        xr = synth(vq, z)
    grid = torchvision.utils.make_grid(xr.cpu(), nrow=images_per_row if images_per_row else nb_repeats)
    TF.to_pil_image(grid).save(out_path)

@torch.no_grad()
def evaluate(
    model_path, 
    data_path, *, 
    batch_size:int=None, 
    out_folder=None, 
    clip_threshold=25, 
    nb_test:int=None, 
    save_images=False, 
    img_folder=None, 
    images_per_row=8, 
    seed=42, 
    clip_model="ViT-B/32",
    compute_fid=False,
    inception_features_real_path:str=None,
    prior_path:str=None,
):
    """
    Evaluate the CLIP score of a model on a dataset of prompts.
    It also optionally saves the generated images of the prompts.
    Also, can optionally compute FID on the generated images using
    the library `piq` (https://github.com/photosynthesis-team/piq).

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
    
    clip_model: str
        CLIP model to use for evaluation
    
    compute_fid: bool
        whether to compute FID.
        Needs the library ``.
        Uses `inception_features_real_path` to get real data features.
        Needs  PIQ library (https://github.com/photosynthesis-team/piq)

    inception_features_real_path: str
        path where real data features are stored for FID.
        Only used if `compute_fid` is True.
    
    prior_path: str
        Use prior for evaluation
    """
 
    name = os.path.basename(data_path) + "_" + clip_model.replace("/", "_")
    if not out_folder:
        out_folder = os.path.dirname(model_path)
        os.makedirs(out_folder, exist_ok=True)
    if not img_folder:
        img_folder = os.path.join(os.path.dirname(model_path), f"eval_{name}_images")
        os.makedirs(img_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_fid:
        from piq.feature_extractors import InceptionV3
        assert inception_features_real_path
        inceptionv3 = InceptionV3().to(device)
        inception_features = []
    
    net = load_model(model_path)
    net.to(device)
    config = net.config
    vqgan_config = config.vqgan_config 
    vqgan_checkpoint = config.vqgan_checkpoint
    clip_size = CLIP_SIZE[clip_model]
    if prior_path:
        prior = load_prior_model(prior_path).to(device)

    perceptor = load_clip_model(clip_model)
    perceptor = perceptor.to(device)

    encoder = load_clip_model(config.clip_model, path=config.get("clip_model_path"))
    encoder = encoder.to(device)

    vq = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    z_min = vq.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = vq.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    mean = torch.Tensor(CLIP_MEAN).view(1,-1,1,1).to(device)
    std = torch.Tensor(CLIP_STD).view(1,-1,1,1).to(device)
    
    toks = load_dataset(data_path)
    if not batch_size:
        batch_size = config.batch_size
        
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if nb_test:
        inds = np.arange(len(toks))
        np.random.shuffle(inds)
        inds = inds[:nb_test]
        toks = toks[inds]
    print(f"Evaluate on {len(toks)} prompts...")
    dataset = torch.utils.data.TensorDataset(toks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    clip_scores_batches = []
    noise_dim = config.noise_dim
    logits_scale = perceptor.logit_scale.exp().to(device)
    normalize_input = config.get("normalize_input", False)
    for batch_idx, (tok,) in enumerate(dataloader):
        tok = tok.to(device)
        H = encoder.encode_text(tok).float()
        if prior_path:
            H = H.view(len(H), -1, 1, 1)
            H = prior.sample(H)
            H = H.view(len(H), -1)
        if normalize_input:
            H = F.normalize(H, dim=1)
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
        xr = synth(vq, z)
        if compute_fid:
            f = inceptionv3(xr)
            f = f[0].view(len(xr), -1)
            f = f.cpu()
            inception_features.append(f)
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

        text_features = perceptor.encode_text(tok).float().to(device)
        text_features = F.normalize(text_features, dim=1) 
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
    if compute_fid:
        from piq import FID
        fake_features = torch.cat(inception_features)
        real_features = torch.load(inception_features_real_path, map_location="cpu")
        fid_metric = FID()
        fid = fid_metric(real_features, fake_features).item()
        fid_dataset = os.path.basename(inception_features_real_path)
        dump[f"fid_{fid_dataset}"] = fid
        print(f"FID: {fid}")

    out = os.path.join(out_folder, f"eval_{name}.json")
    print(f"Saving to {out}")
    with open(out, "w") as fd:
        fd.write(json.dumps(dump))
    print(f"CLIP score mean: {mean}. CLIP score std:{std}")
    print(f"Fraction of images with a CLIP score of at least {clip_threshold}: {clip_score_atleast}")
    return dump

def load_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict):
        # checkpoint, preferred way to load models
        config = ckpt['config']
        net = build_model(config)
        net.load_state_dict(ckpt['state_dict'])
        net.config = config
    else:
        # backward compatibility
        net = ckpt
        config = net.config
        if net.config.model_type == "mlp_mixer":
            _fix_mlp_mixer_gelu_issue(net)
        elif net.config.model_type == "vitgan":
            _fix_vitgan_gelu_issue(net)
            
    return net


def load_dataset(path):
    # path can be the following:
    # - a path to a text file where each line is a text prompt
    # - a glob pattern (*) of text files where each file is a text prompt
    # - a pkl file created using `tokenize` or `encode_images` or `encode_text_and_images`
    if path.endswith("pkl"):
        toks = torch.load(path)
    elif "*" in path:
        texts = [open(f).read().strip() for f in glob(path)]
        toks = clip.tokenize(texts, truncate=True)
    else:
        texts = [t.strip() for t in open(path).readlines()]
        toks = clip.tokenize(texts, truncate=True)
    return toks

def load_clip_model(model_type, path=None):
    if model_type in ("cloob_rn50", "cloob_rn50x4"):
        # CLOOB ckpts from original paper
        perceptor = CLOOB(path=path).eval().requires_grad_(False)
    elif model_type in ("cloob_laion_400m_vit_b_16_16_epochs", "cloob_laion_400m_vit_b_16_32_epochs"):
        # CLOOB models trained by @crowsonkb on LAION
        import cloob_crowsonkb
        config = cloob_crowsonkb.get_config(model_type)
        model = cloob_crowsonkb.get_pt_model(config)
        checkpoint = cloob_crowsonkb.download_checkpoint(config)
        model.load_state_dict(cloob_crowsonkb.get_pt_params(config, checkpoint))
        model.eval().requires_grad_(False)
        perceptor = model
        perceptor.encode_image = perceptor.image_encoder
        perceptor.encode_text = perceptor.text_encoder
    elif "openclip" in model_type:
        # OpenCLIP models trained on LAION-400M, LAION-2B
        #e.g., openclip/ViT-B-32-quickgelu/laion400m_e32
        import open_clip
        prefix, arch, dataset = model_type.split("/")
        perceptor, train_transform, eval_transform = open_clip.create_model_and_transforms(arch, pretrained=dataset)
        perceptor = perceptor.eval().requires_grad_(False)
    else:
        # OpenAI CLIP
        perceptor = clip.load(model_type, jit=False)[0].eval().requires_grad_(False)
    return perceptor

def train_prior(config_path):
    from net2net.modules.flow.flatflow import ConditionalFlatCouplingFlow
    from net2net.modules.flow.loss import NLL
    use_horovod = USE_HOROVOD
    if use_horovod:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
    config = OmegaConf.load(config_path)
    config.folder = os.path.dirname(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isdir(config.data.path):
        paths = glob(os.path.join(config.data.path, "*"))
        paths = sorted(paths)
        random.shuffle(paths)
        if use_horovod:
            paths = [p for i, p in enumerate(paths) if i % hvd.size() == hvd.rank()]
        xs = []
        ys = []
        for p in paths:
            x, y = torch.load(p)
            xs.append(x)
            ys.append(y)
        x = torch.cat(xs)
        y = torch.cat(ys)
    else:
        x,y = torch.load(config.data.path)
    hvd.join()
    dataset = torch.utils.data.TensorDataset(x,y)
    if use_horovod:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=hvd.size(),
            rank=hvd.rank(),
        )
        shuffle =  None
    else:
        sampler = None
        shuffle = True
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.data.batch_size,  
        shuffle=shuffle,
        sampler=sampler,
    )

    input_size = x.shape[1]
    output_size  = y.shape[1]
    opt_path = os.path.join(config.folder, "opt.th")
    checkpoint_path = os.path.join(config.folder, "checkpoint.th")
    if os.path.exists(checkpoint_path):
        print("Resuming")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        step = ckpt['step']
        flow = build_prior_model(config, input_size, output_size)
        flow.load_state_dict(ckpt['model'])
    else:
        step = 0
        flow = build_prior_model(config, input_size, output_size)
    if os.path.exists(opt_path):
        opt_state_dict = torch.load(opt_path, map_location="cpu")
    else:
        opt_state_dict = None
    flow = flow.to(device)
    get_loss = NLL()
    params = flow.parameters()
    clip_grad_norm = config.optim.get("clip_grad_norm")
    opt = torch.optim.Adam(
        params,
        lr=config.optim.lr 
    )
    if opt_state_dict:
        opt.load_state_dict(opt_state_dict)
    if use_horovod:
        hvd.broadcast_parameters(flow.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(opt, root_rank=0)
    rank_zero = (use_horovod and hvd.rank() == 0) or not use_horovod
    if rank_zero:
        writer = SummaryWriter(config.folder)
    for epoch in range(config.optim.epochs):
        if use_horovod:
            sampler.set_epoch(epoch)
        for inp, out in dataloader:
            bs = len(inp)
            inp = inp.to(device)
            out = out.to(device)
            inp = inp.view(bs, -1, 1, 1)
            out = out.view(bs, -1, 1, 1)
            zz, logdet = flow(out, inp)
            loss, log_dict = get_loss(zz, logdet)
            opt.zero_grad()
            loss.backward()
            if clip_grad_norm:
                clip_grad_norm_(flow.parameters(), clip_grad_norm)
            opt.step()
            if step % 100 == 0 and rank_zero:
                for k, v in log_dict.items():
                    writer.add_scalar(k, v.item(), step)
            if step % config.logging.log_interval == 0 and rank_zero:
                print(epoch, step, loss.item())
                print(log_dict)
                ckpt = {
                    "model": flow.state_dict(),
                    "step": step,
                    "input_size": input_size,
                    "output_size": output_size,
                    "config": config,
                }
                torch.save(ckpt, checkpoint_path)
                torch.save(opt.state_dict(), opt_path)
            step += 1

def load_prior_model(prior_path):
    ckpt = torch.load(prior_path, map_location='cpu')
    prior = build_prior_model(ckpt['config'], ckpt['input_size'], ckpt['output_size'])
    prior.load_state_dict(ckpt['model'])
    return prior

def build_prior_model(config, input_size, output_size):
    from net2net.modules.flow.flatflow import ConditionalFlatCouplingFlow
    return ConditionalFlatCouplingFlow(
        in_channels=output_size, 
        conditioning_dim=input_size, 
        embedding_dim=config.model.embedding_dim, 
        hidden_dim=config.model.hidden_dim,
        hidden_depth=config.model.hidden_depth,
        n_flows=config.model.n_flows,
    )

if __name__ == "__main__":
    run([
        train, 
        test, 
        tokenize, 
        encode_text_and_images, 
        encode_text_and_images_webdataset, 
        evaluate,
        train_prior,
    ])
