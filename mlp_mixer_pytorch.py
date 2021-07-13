import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(image_size, channels, patch_size, dim, depth, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
    )

class Mixer(torch.nn.Module):

    def __init__(self, input_dim, image_size, channels, patch_size, dim, depth, expansion_factor = 4, dropout = 0., nb_gen=1):
        super().__init__()
        self.mixer = MLPMixer(image_size, channels, patch_size, dim, depth, expansion_factor=expansion_factor, dropout=dropout)
        self.proj = nn.Linear(input_dim, nb_gen*image_size*image_size*channels)
        self.final_proj = nn.Linear(dim, channels)
        self.channels = channels
        self.image_size = image_size
        self.nb_gen = nb_gen
    
    def forward(self, x):
        bs = len(x)
        x = self.proj(x)
        x = x.view(bs*self.nb_gen, self.channels, self.image_size, self.image_size)
        x = self.mixer(x)
        x = self.final_proj(x)
        x = x.view(bs*self.nb_gen, self.image_size, self.image_size, self.channels)
        x = x.permute(0,3,1,2,)
        return x

# import torch
# dim = 128
# net = Mixer(input_dim=128, image_size=32, channels=3, patch_size=1, dim=dim, depth=8)
# x = torch.randn(1,128)
# y = net(x)
# print(y.shape)
