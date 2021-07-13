from collections import defaultdict
import shutil
from datetime import datetime
import yaml
from glob import glob
import joblib
from argparse import Namespace
import time
import os
import sys
import math
import json
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
from clize import run
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

# from adamp import AdamP
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms

from PIL import Image
from pathlib import Path

# assert torch.cuda.is_available(), "You need to have an Nvidia GPU with CUDA installed."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# constants
EPS = 1e-8

# helper classes


class NanException(Exception):
    pass


class EMA:
    """
    EM stands for exponential moving average.
    The way it works is that the model weights
    are cloned in a separated model instance,
    and the weights of that separated model
    are the exponential moving average of the
    weights of the model that is being trained.
    This trick make the generated samples better in general.
    However, it does not affect training in any way, it just
    affects the generated samples at evaluation.
    See " Progressive growing of GANs forimproved quality, stability, and variation"
    for the original reference.
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
    """
    Just flattens any n-dim tensor into a matrix
    """

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss


# one layer of self-attention and feedforward, for images

# attn_and_ff = lambda chan: nn.Sequential(*[
# Residual(Rezero(ImageLinearAttention(chan))),
# Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
# ])

# helpers


def default(value, d):
    return d if value is None else value


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def cast_list(el):
    return el if isinstance(el, list) else [el]


def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return t is None


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
            optimizer.synchronize()
    else:
        loss.backward(**kwargs)


def gradient_penalty(images, output, weight=10):
    """
    Gradient penalty used in Improved Training of Wasserstein
    GAN paper.
    The norm of the gradient of the output of the discr
    with respect to the images is penalized.
    """
    batch_size = images.shape[0]
    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size()).to(DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # gradients = gradients.view(batch_size, -1)
    gradients = gradients.reshape((batch_size, -1))
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    """
    pl stands for Path Length.
    This is the path length regularization used
    in StyleGAN2 paper. See section 3.2 of the paper
    """
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).to(DEVICE) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(
        outputs=outputs,
        inputs=styles,
        grad_outputs=torch.ones(outputs.shape).to(DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()


def noise(n, latent_dim):
    return torch.randn(n, latent_dim)


def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]


def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)


def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0.0, 1.0)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i).cpu() for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def evaluate_func_in_chunks(X, func, batch_size):
    ys = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        y = func(batch).cpu()
        ys.append(y)
    return torch.cat(ys)


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high
    return res


# dataset


def convert_rgb_to_transparent(image):
    if image.mode == "RGB":
        return image.convert("RGBA")
    return image


def convert_transparent_to_rgb(image):
    if image.mode == "RGBA":
        return image.convert("RGB")
    return image


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f"image with invalid number of channels given {channels}")

        if alpha is None and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


# augmentations


def random_float(lo, hi):
    return lo + (hi - lo) * random()


def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random() * delta)
    w_delta = int(random() * delta)
    cropped = tensor[
        :, :, h_delta : (h_delta + new_width), w_delta : (w_delta + new_width)
    ].clone()
    return F.interpolate(cropped, size=(h, h), mode="bilinear")


def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    # from lucidrains original code
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0.0, detach=False):
        if random() < prob:
            random_scale = random_float(0.75, 0.95)
            images = random_hflip(images, prob=0.5)
            images = random_crop_and_resize(images, scale=random_scale)

        if detach:
            images.detach_()

        return self.D(images)


class DiffAugWrapper(nn.Module):
    # from diffaug paper https://arxiv.org/pdf/2006.10738.pdf
    def __init__(self, D, policy):
        super().__init__()
        self.D = D
        self.policy = policy

    def forward(self, images, prob=None, detach=False):
        images = DiffAugment(images, policy=self.policy)
        if detach:
            images.detach_()
        return self.D(images)


# stylegan2 classes


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class RGBBlock(nn.Module):
    """
    The RGBBlock correspond to  Figure 7 b) top in StyleGAN2 paper.
    There is one RGBBlock for each resolution, and each RGBBlock
    of a resolution takes as input the RGBBlock of the previous resolution.
    Using this block, the full res image pixels are constructed progressively
    without needing for explicit progressive training like in ProGAN.
    """

    def __init__(self, latent_dim, input_channel, upsample, rgba=False, out_filters=1):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsample
            else None
        )

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        # if prev_rgb is not None:
            # x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class Conv2DMod(nn.Module):
    """
    This layer is used each time we need to fuse
    information from some conv feature maps and
    a style vector. This layer is basically a replacement
    for AdaIN that was used in StyleGAN1 and they found
    worked better.
    """

    def __init__(
        self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs
    ):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(
            self.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
        )

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlock(nn.Module):
    """
    Basic building block of a StyleGAN2 generator.
    Each GeneratorBlock use information from the style
    and the previous block and the noise and multiplies
    the resolution by 2
    """

    def __init__(
        self,
        latent_dim,
        input_channels,
        filters,
        upsample=True,
        upsample_rgb=True,
        rgba=False,
        nb_channels=1,
    ):
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsample
            else None
        )

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba, nb_channels)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, : x.shape[2], : x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))
        
        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class DiscriminatorBlock(nn.Module):
    """
    Basic building block of the StyleGAN2 discriminator
    """

    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(
            input_channels, filters, 1, stride=(2 if downsample else 1)
        )

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        self.downsample = (
            nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class Generator(nn.Module):
    """
    a StyleGAN2 generator is bascially a composition
    of several GeneratorBlocks
    """

    def __init__(
        self,
        image_size,
        latent_dim,
        network_capacity=16,
        transparent=False,
        attn_layers=[],
        no_const=False,
        fmap_max=512,
        nb_channels=1,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][
            ::-1
        ]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(
                latent_dim, init_channels, 4, 1, 0, bias=False
            )
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent,
                nb_channels=nb_channels,
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb


class Discriminator(nn.Module):
    """
    A StyleGAN2 Discriminator is a composition of DiscriminatorBlocks
    """

    def __init__(
        self,
        image_size,
        network_capacity=16,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        transparent=False,
        fmap_max=512,
        nb_channels=1,
    ):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        # num_init_filters = 3 if not transparent else 4
        num_init_filters = nb_channels
        blocks = []
        filters = [num_init_filters] + [(network_capacity*4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = (
                PermuteToFrom(VectorQuantize(out_chan, fq_dict_size))
                if num_layer in fq_layers
                else None
            )
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(
            self.blocks, self.attn_blocks, self.quantize_blocks
        ):
            x = block(x)

            if attn_block is not None:
                x = attn_block(x)

            if q_block is not None:
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss


class StyleGAN2(nn.Module):
    """
    Contains everything needed for training:

    - generator, style vectorizer and their optimizer
    - EMA (exponential moving average) versions  of the generator and the stylevectorizer
    - discriminator and its optimizer
    """

    def __init__(
        self,
        image_size,
        latent_dim=512,
        fmap_max=512,
        style_depth=8,
        network_capacity=16,
        transparent=False,
        fp16=False,
        cl_reg=False,
        steps=1,
        lr=1e-4,
        ttur_mult=2,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        no_const=False,
        lr_mlp=0.1,
        nb_channels=1,
        diffaug_policy=None,
        adam_eps=1e-8,
    ):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.G = Generator(
            image_size,
            latent_dim,
            network_capacity,
            transparent=transparent,
            attn_layers=attn_layers,
            no_const=no_const,
            fmap_max=fmap_max,
            nb_channels=nb_channels,
        )
        self.D = Discriminator(
            image_size,
            network_capacity,
            fq_layers=fq_layers,
            fq_dict_size=fq_dict_size,
            attn_layers=attn_layers,
            transparent=transparent,
            fmap_max=fmap_max,
            nb_channels=nb_channels,
        )
        # Exponential moving average versions of the style vectorizer
        # and the generator. Only eaffects evaluatio/test time generation
        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.GE = Generator(
            image_size,
            latent_dim,
            network_capacity,
            transparent=transparent,
            attn_layers=attn_layers,
            no_const=no_const,
            nb_channels=nb_channels,
        )

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner

            # experimental contrastive loss discriminator regularization
            assert (
                not transparent
            ), "contrastive loss regularization does not work with transparent images yet"
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer="flatten")

        # er for augmenting all images going into the discriminator
        if diffaug_policy:
            self.D_aug = DiffAugWrapper(self.D, diffaug_policy)
        else:
            self.D_aug = AugWrapper(self.D, image_size)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
            
        cls = AdamP
        self.G_opt = cls(generator_params, lr=self.lr, betas=(0.5, 0.9), eps=adam_eps)
        self.D_opt = cls(
            self.D.parameters(), lr=self.lr * ttur_mult, betas=(0.5, 0.9), eps=adam_eps
        )
        self.G_opt = hvd.DistributedOptimizer(
            self.G_opt,
            # backward_passes_per_step=self.gradient_accumulate_every,
        )
        self.D_opt = hvd.DistributedOptimizer(
            self.D_opt,
            # backward_passes_per_step=self.gradient_accumulate_every,
        )
        self._init_weights()
        self.reset_parameter_averaging()

        self.to(DEVICE)

        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (
                self.G_opt,
                self.D_opt,
            ) = amp.initialize(
                [self.S, self.G, self.D, self.SE, self.GE],
                [self.G_opt, self.D_opt],
                opt_level="O2",
                num_losses=3,
            )

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(
                current_model.parameters(), ma_model.parameters()
            ):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x


class SG(nn.Module):

    def __init__(self, image_size=256, latent_dim=512, nb_channels=512, depth=8, truncate=256, network_capacity=16):
        super().__init__()
        self.gen = Generator(image_size=image_size, latent_dim=latent_dim, nb_channels=nb_channels, network_capacity=network_capacity)
        self.mapping = StyleVectorizer(latent_dim, depth)
        self.image_size = image_size
        self.truncate = truncate

    def forward(self, z, noise=None):
        if noise is None:
            noise = torch.zeros(len(z), self.image_size, self.image_size, 1).to(z.device)
        w = self.mapping(z)
        w = w.view(len(w), 1, w.shape[1]).repeat(1, 9, 1)
        x = self.gen(w, noise)
        x = x[:, :, :self.truncate, :self.truncate]
        return x


class SGDummy(nn.Module):

    def __init__(self, image_size=256, latent_dim=512, nb_channels=512, depth=8, truncate=256, network_capacity=16):
        super().__init__()
        H = 4096
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, H),
            nn.LayerNorm(H),
            leaky_relu(),
            nn.Linear(H, H),
            nn.LayerNorm(H),
            leaky_relu(),
            nn.Linear(H, H),
            nn.LayerNorm(H),
            leaky_relu(),
            nn.Linear(H, H),
            nn.LayerNorm(H),
            leaky_relu(),
            nn.Linear(H, image_size*image_size*nb_channels),
        )
        self.nb_channels = nb_channels
        self.image_size = image_size

    def forward(self, z, noise=None):
        x = self.fc(z)
        x = x.view(len(x), self.nb_channels, self.image_size, self.image_size)
        return x


if __name__ == "__main__":
    g = SG(
        image_size=512,
        latent_dim=512,
        nb_channels=512,
    )
    z = torch.randn(1, 512)
    x = g(z)
    print(x.shape)
