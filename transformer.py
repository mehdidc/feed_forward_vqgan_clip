import torch.nn as nn
import torch
from x_transformers import ContinuousTransformerWrapper, Decoder

class XTransformer(nn.Module):
    
    def __init__(self, input_dim, image_size, channels, dim, depth, heads):
        super().__init__()
        self.input_dim = input_dim
        self.transformer = model = ContinuousTransformerWrapper(
            dim_in = dim,
            dim_out = channels,
            max_seq_len = image_size*image_size,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                heads = heads
            )
        ) 
        self.dim = dim
        self.proj = nn.Linear(input_dim, image_size*image_size*dim)
        self.channels = channels
        self.image_size = image_size
    
    def forward(self, x):
        bs = len(x)
        x = self.proj(x)
        x = x.view(bs, self.image_size*self.image_size, self.dim)
        x = self.transformer(x)
        x = x.view(bs, self.image_size, self.image_size, self.channels)
        x = x.permute(0,3,1,2,)
        return x
