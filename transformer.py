import torch.nn as nn
import torch
from x_transformers import ContinuousTransformerWrapper, Decoder

class XTransformer(nn.Module):
    
    def __init__(self, input_dim, image_size, channels, dim, depth, heads, initial_proj=True, add_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.add_input = add_input
        self.transformer = model = ContinuousTransformerWrapper(
            dim_in = dim if initial_proj else input_dim,
            dim_out = channels,
            max_seq_len = image_size*image_size + (0 if self.add_input else 1),
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                heads = heads
            )
        ) 
        self.dim = dim
        if initial_proj:
            self.proj = nn.Linear(input_dim, image_size*image_size*dim)
        self.initial_proj = initial_proj
        self.channels = channels
        self.image_size = image_size

    def forward(self, x):
        bs = len(x)
        if hasattr(self, "proj"):
            x = self.proj(x)
            x = x.view(bs, self.image_size*self.image_size, self.dim)
        else:
            if self.add_input:
                x = x.view(bs, 1, self.input_dim)
                x = x.repeat(1, self.image_size**2, 1)
            else:
                inp = torch.zeros(bs, self.image_size**2, self.input_dim).to(x.device)    
                x = x.view(bs, 1, self.input_dim)
                x = torch.cat((x, inp), dim=1)
        x = self.transformer(x)
        if not hasattr(self, "proj") and not self.add_input:
            x = x[:, 1:]
        x = x.view(bs, self.image_size, self.image_size, self.channels)
        x = x.permute(0,3,1,2,)
        return x

if __name__ == '__main__':

    net = XTransformer(
        input_dim=512, image_size=16, channels=256, dim=256, 
        depth=8, heads=8, initial_proj=False, 
        add_input=False
    )
    x=torch.randn(2,512)
    y = net(x)
    print(y[0,:,0,0])
    print(y[1,:,0,0])
    
