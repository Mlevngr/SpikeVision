import torch
import torch.nn as nn
import sys
sys.path.append('.')

from lif_neuron import LIFNeuron


class Conv(nn.Module):
    def __init__(
            self,
            threshold=128/128,
            in_channels=2,
            embed_dims=256,
            pooling_stat="0011",
            train_threshold=False,

    ):
        super().__init__()
        self.threshold=threshold
        self.pooling_stat = pooling_stat
        self.conv1 = nn.Conv2d(
            in_channels,
            embed_dims // 8, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.mp = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        )
        self.conv2 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv3 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv4 = nn.Conv2d(
            embed_dims // 2,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv5 = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        torch.manual_seed(42)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)

        self.lif_input = LIFNeuron(threshold=3, decay=0, min=0.0, reset=True, name="lif_Conv_input", train=train_threshold)
        self.lif_conv1 = LIFNeuron(threshold=threshold, decay=0.0, min=0.0, reset=True, name="lif_Conv_conv1", train=train_threshold)
        self.lif_conv2 = LIFNeuron(threshold=threshold, decay=0.0, min=0.0, reset=True, name="lif_Conv_conv2", train=train_threshold)
        self.lif_conv3 = LIFNeuron(threshold=threshold, decay=0.0, min=0.0, reset=True, name="lif_Conv_conv3", train=train_threshold)
        self.lif_conv4 = LIFNeuron(threshold=threshold, decay=0.0, min=0.0, reset=True, name="lif_Conv_conv4", train=train_threshold)
        self.lif_conv5 = LIFNeuron(threshold=threshold, decay=0.0, min=0.0, reset=True, name="lif_Conv_conv5", train=train_threshold)

    def forward(self, x, threshold):
        B, _, H, W = x.shape
        ratio = 1
        x = self.lif_input(x, threshold=threshold)
        x = x
        x = self.conv1(x)#.reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.lif_conv1(x)
        if self.pooling_stat[0] == "1":
            x = self.mp(x)
            ratio *= 2

        x = self.conv2(x)#.reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.lif_conv2(x)
        if self.pooling_stat[1] == "1":
            x = self.mp(x)
            ratio *= 2 

        x = self.conv3(x)#.reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.lif_conv3(x)
        if self.pooling_stat[2] == "1":
            x = self.mp(x)
            ratio *= 2

        x = self.conv4(x)
        x = x#.reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.lif_conv4(x)

        if self.pooling_stat[3] == "1":
            x = self.mp(x)
            ratio *= 2

        x_res = x.clone()
        x = self.conv5(x)
        x = (0.5*(x + x_res)).reshape(B, -1, H // ratio, W // ratio).contiguous()

        return x
    


