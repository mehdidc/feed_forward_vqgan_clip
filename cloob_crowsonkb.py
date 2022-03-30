# CLOOB code from https://github.com/crowsonkb/cloob-training, thanks to @crowsonkb
import math
import pickle

import clip
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from hashlib import sha256
from pathlib import Path
from torch import hub
import json

class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([seq_len, d_model]))

    def forward(self, x):
        return x + self.weight


class GELU(nn.Module):
    def __init__(self, approximate=True):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate:
            return x * (1 + torch.tanh((2 / math.pi)**0.5 * (x + 0.047715 * x**3))) / 2
        return F.gelu(x)


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        assert d_model % n_heads == 0
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, x, padding_mask=None):
        n, s, d = x.shape
        head_size = d // self.n_heads
        x = self.norm(x)
        q = self.query(x).view([*x.shape[:-1], self.n_heads, head_size])
        k = self.key(x).view([*x.shape[:-1], self.n_heads, head_size])
        v = self.value(x).view([*x.shape[:-1], self.n_heads, head_size])
        attn_logits = torch.einsum('...thd,...mhd->...htm', q, k) / head_size**0.5
        if padding_mask is not None:
            mask = padding_mask[:, None, :, None]
            attn_logits = torch.where(mask, attn_logits, attn_logits.new_tensor(-1e30))
        attn_weights = attn_logits.softmax(-1)
        attn = torch.einsum('...htm,...mhd->...thd', attn_weights, v)
        attn_vec = attn.reshape(x.shape)
        return self.out(attn_vec)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear_0 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.linear_1 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear_0(x)
        x = self.act(x)
        x = self.linear_1(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)

    def __call__(self, x, padding_mask=None):
        x = x + self.attn(x, padding_mask)
        x = x + self.ff(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, d_embed, n_layers, d_model, seq_len, n_heads, vocab_size):
        super().__init__()
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = self.d_model * 4
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.eot_token = vocab_size - 1
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embed = PositionalEmbedding(self.seq_len, self.d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(self.d_model, self.d_ff, self.n_heads) for _ in range(n_layers)])
        self.proj = nn.Linear(self.d_model, self.d_embed)

    def forward(self, x):
        eot_mask = x == self.eot_token
        padding_mask = (torch.cumsum(eot_mask, dim=-1) == 0) | eot_mask
        x = self.embed(x)
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = x[:, 0]
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        return x


class ViTImageEncoder(nn.Module):
    def __init__(self, d_embed, n_layers, d_model, seq_len, n_heads, input_channels, patch_size):
        super().__init__()
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = self.d_model * 4
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed = nn.Conv2d(self.input_channels, self.d_model, self.patch_size, self.patch_size, bias=False)
        self.class_embed = nn.Parameter(torch.randn([self.d_model]) / self.d_model**0.5)
        self.pos_embed = PositionalEmbedding(self.seq_len + 1, self.d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(self.d_model, self.d_ff, self.n_heads) for _ in range(n_layers)])
        self.proj = nn.Linear(self.d_model, self.d_embed)

    def forward(self, x):
        x = self.embed(x)
        x = x.reshape([x.shape[0], x.shape[1], -1]).permute([0, 2, 1])
        x = torch.cat([self.class_embed[None, None].repeat([x.shape[0], 1, 1]), x], dim=1)
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        return x


class CLOOBModel(nn.Module):
    def __init__(self, config, image_encoder, text_encoder):
        super().__init__()
        self.config = config
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.normalize = transforms.Normalize(self.config['image_encoder']['normalize']['mean'],
                                              self.config['image_encoder']['normalize']['std'])
        assert self.config['text_encoder']['tokenizer'] == 'clip'
        self.tokenize = clip.tokenize


def get_pt_model(config):
    assert config['image_encoder']['type'] == 'ViT'
    assert config['text_encoder']['type'] == 'transformer'
    image_encoder = ViTImageEncoder(
        config['d_embed'],
        config['image_encoder']['n_layers'],
        config['image_encoder']['d_model'],
        (config['image_encoder']['image_size'] // config['image_encoder']['patch_size'])**2,
        config['image_encoder']['n_heads'],
        config['image_encoder']['input_channels'],
        config['image_encoder']['patch_size'],
    )
    text_encoder = TextEncoder(
        config['d_embed'],
        config['text_encoder']['n_layers'],
        config['text_encoder']['d_model'],
        config['text_encoder']['text_size'],
        config['text_encoder']['n_heads'],
        config['text_encoder']['vocab_size'],
    )
    return CLOOBModel(config, image_encoder, text_encoder)


def map_to_tensor(x):
    return {k: {k2: torch.tensor(v2) for k2, v2 in v.items()} for k, v in x.items()}


def convert_jax_vit_image_params(params):
    base = 'vi_t_image_encoder'
    pt_base = 'image_encoder'
    params = map_to_tensor(params)
    state = {}
    for k, v in params.items():
        names = k.split('/')
        if k == base:
            state[f'{pt_base}.class_embed'] = v['class_embed']
        elif names[1] == 'embed':
            state[f'{pt_base}.embed.weight'] = v['w'].permute([3, 2, 0, 1])
        elif names[1] == 'pos_embed':
            state[f'{pt_base}.pos_embed.weight'] = v['w']
        elif names[1] == 'proj':
            state[f'{pt_base}.proj.weight'] = v['w'].T
            state[f'{pt_base}.proj.bias'] = v['b']
        elif names[1].startswith('layer'):
            layer_num = int(names[1].partition('_')[2])
            if names[2] == 'self_attention':
                if names[3] == 'layer_norm':
                    state[f'{pt_base}.layers.{layer_num}.attn.norm.weight'] = v['scale']
                    state[f'{pt_base}.layers.{layer_num}.attn.norm.bias'] = v['offset']
                elif names[3] == 'multi_head_attention':
                    if names[4] == 'query':
                        state[f'{pt_base}.layers.{layer_num}.attn.query.weight'] = v['w'].T
                        state[f'{pt_base}.layers.{layer_num}.attn.query.bias'] = v['b']
                    elif names[4] == 'key':
                        state[f'{pt_base}.layers.{layer_num}.attn.key.weight'] = v['w'].T
                        state[f'{pt_base}.layers.{layer_num}.attn.key.bias'] = v['b']
                    elif names[4] == 'value':
                        state[f'{pt_base}.layers.{layer_num}.attn.value.weight'] = v['w'].T
                        state[f'{pt_base}.layers.{layer_num}.attn.value.bias'] = v['b']
                    elif names[4] == 'linear':
                        state[f'{pt_base}.layers.{layer_num}.attn.out.weight'] = v['w'].T
                        state[f'{pt_base}.layers.{layer_num}.attn.out.bias'] = v['b']
            elif names[2] == 'feed_forward':
                if names[3] == 'layer_norm':
                    state[f'{pt_base}.layers.{layer_num}.ff.norm.weight'] = v['scale']
                    state[f'{pt_base}.layers.{layer_num}.ff.norm.bias'] = v['offset']
                elif names[3] == 'linear_0':
                    state[f'{pt_base}.layers.{layer_num}.ff.linear_0.weight'] = v['w'].T
                    state[f'{pt_base}.layers.{layer_num}.ff.linear_0.bias'] = v['b']
                elif names[3] == 'linear_1':
                    state[f'{pt_base}.layers.{layer_num}.ff.linear_1.weight'] = v['w'].T
                    state[f'{pt_base}.layers.{layer_num}.ff.linear_1.bias'] = v['b']
    return state


def convert_jax_text_params(params):
    base = 'text_encoder'
    pt_base = 'text_encoder'
    params = map_to_tensor(params)
    state = {}
    for k, v in params.items():
        names = k.split('/')
        if names[1] == 'embed':
            state[f'{pt_base}.embed.weight'] = v['embeddings']
        elif names[1] == 'pos_embed':
            state[f'{pt_base}.pos_embed.weight'] = v['w']
        elif names[1] == 'proj':
            state[f'{pt_base}.proj.weight'] = v['w'].T
            state[f'{pt_base}.proj.bias'] = v['b']
        elif names[1].startswith('layer'):
            layer_num = int(names[1].partition('_')[2])
            if names[2] == 'self_attention':
                if names[3] == 'layer_norm':
                    state[f'{pt_base}.layers.{layer_num}.attn.norm.weight'] = v['scale']
                    state[f'{pt_base}.layers.{layer_num}.attn.norm.bias'] = v['offset']
                elif names[3] == 'multi_head_attention':
                    if names[4] == 'query':
                        state[f'{pt_base}.layers.{layer_num}.attn.query.weight'] = v['w'].T
                        state[f'{pt_base}.layers.{layer_num}.attn.query.bias'] = v['b']
                    elif names[4] == 'key':
                        state[f'{pt_base}.layers.{layer_num}.attn.key.weight'] = v['w'].T
                        state[f'{pt_base}.layers.{layer_num}.attn.key.bias'] = v['b']
                    elif names[4] == 'value':
                        state[f'{pt_base}.layers.{layer_num}.attn.value.weight'] = v['w'].T
                        state[f'{pt_base}.layers.{layer_num}.attn.value.bias'] = v['b']
                    elif names[4] == 'linear':
                        state[f'{pt_base}.layers.{layer_num}.attn.out.weight'] = v['w'].T
                        state[f'{pt_base}.layers.{layer_num}.attn.out.bias'] = v['b']
            elif names[2] == 'feed_forward':
                if names[3] == 'layer_norm':
                    state[f'{pt_base}.layers.{layer_num}.ff.norm.weight'] = v['scale']
                    state[f'{pt_base}.layers.{layer_num}.ff.norm.bias'] = v['offset']
                elif names[3] == 'linear_0':
                    state[f'{pt_base}.layers.{layer_num}.ff.linear_0.weight'] = v['w'].T
                    state[f'{pt_base}.layers.{layer_num}.ff.linear_0.bias'] = v['b']
                elif names[3] == 'linear_1':
                    state[f'{pt_base}.layers.{layer_num}.ff.linear_1.weight'] = v['w'].T
                    state[f'{pt_base}.layers.{layer_num}.ff.linear_1.bias'] = v['b']
    return state


def get_pt_params(config, checkpoint):
    assert config['image_encoder']['type'] == 'ViT'
    assert config['text_encoder']['type'] == 'transformer'
    cloob_params = pickle.load(open(checkpoint, 'rb'))['params']
    state = {**convert_jax_vit_image_params(cloob_params[0]), **convert_jax_text_params(cloob_params[1])}
    return state

def load_config(path):
    return json.load(open(path))


def list_configs():
    return sorted(path.stem for path in glob("*.json"))


def get_config(name):
    config_path =  Path(name + '.json')
    if config_path.is_file():
        return load_config(config_path)


def download_checkpoint(config):
    dest_dir = Path(hub.get_dir()) / 'cloob'
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = config['url']
    filename = url.rpartition('/')[2]
    model_hash = filename.rpartition('-')[2].partition('.')[0]
    dest_file = dest_dir / filename
    if not dest_file.exists():
        hub.download_url_to_file(url, str(dest_file), hash_prefix=model_hash)
    if sha256(open(dest_file, 'rb').read()).hexdigest() == model_hash:
        return str(dest_file)
    raise RuntimeError(f'Model has been downloaded to {dest_file!s} but the SHA256 checksum does not not match')
