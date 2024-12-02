import torch
import torch.nn as nn
import sys
sys.path.append('.')

from lif_neuron import LIFNeuron
    
class SSA(nn.Module):
    def __init__(
        self,
        embed_dim,
        threshold=128/128,
        layer=0,
        train_threshold=False,
    ):
        super().__init__()
        self.threshold = threshold
        self.embed_dim = embed_dim
        self.sieve1_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.sieve2_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.source_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.lin_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)

        self.lif_input = LIFNeuron(threshold=threshold, decay=0.5, reset=True, name="lif_enc_input", train=train_threshold)
        self.lif_source = LIFNeuron(threshold=threshold, decay=0.5, min=0, reset=True, name="lif_v", train=train_threshold)
        self.lif_Sieve = LIFNeuron(threshold=4.0, decay=0.5, reset=True, name="lif_kv", train=True)
        self.lif_lin = LIFNeuron(threshold=1.0, decay=0.5, reset=True, name="lif_proj", train=train_threshold)
        torch.manual_seed(42)
        nn.init.xavier_normal_(self.sieve1_conv.weight)
        nn.init.xavier_normal_(self.sieve2_conv.weight)
        nn.init.xavier_normal_(self.source_conv.weight)
        nn.init.xavier_normal_(self.lin_conv.weight)
        self.layer = layer
    
    def forward(self, x):
        B, C, H, W = x.shape

        x = self.lif_input(x, threshold=1)
        x_res = x.clone()

        x_input = x
        source = self.source_conv(x_input)
        source = self.lif_source(source)

        sieve1 = self.sieve1_conv(x_input)
        sieve2 = self.sieve2_conv(x_input)
        Sieve = sieve1.add(sieve2)
        Sieve = self.lif_Sieve(Sieve)

        x = Sieve.mul(source)

        x = self.lin_conv(x).contiguous()


        x += x_res
        x *= 0.5

        return x
    
class Screener(nn.Module):
    def __init__(
        self,
        dim,
        threshold=128/128,
        layer=0,
        train_threshold=True,
    ):
        super().__init__()
        self.attn = SSA(
            threshold=threshold,
            embed_dim=dim,
            layer=layer,
            train_threshold=train_threshold,
        )
    
    def forward(self, x):
        x = self.attn(x)
        return x

