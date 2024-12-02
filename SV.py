import torch
import torch.nn as nn
from timm.data import create_loader
from lif_neuron import LIFNeuron
from timm.models.registry import register_model
from torchvision.datasets.mnist import MNIST

from conv import Conv
from screener import Screener

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpikeVision(nn.Module):
    def __init__(
        self,
        dataset="DVS128",
        image_size_h=128,
        image_size_w=128,
        input_channels=2,
        num_classes=10,
        embed_dims=256,
        threshold_head=128/128,
        threshold_conv=128/128,
        threshold_scre=128/128,
        depths=1,
        pooling_state="0011",
        train_threshold=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.threshold = threshold_head
        self.num_classes = num_classes
        self.H_temp : int = image_size_h//(2**(pooling_state.count('1')))
        self.W_temp : int = image_size_w//(2**(pooling_state.count('1')))

        self.depths = depths
        self.embed_dims = embed_dims
        
        torch.manual_seed(42)

        self.conv = Conv(
            threshold=threshold_conv,
            in_channels=input_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_state,
            train_threshold=train_threshold,
        )

        self.blocks = nn.ModuleList(
            [
                Screener(
                    dim=embed_dims,
                    threshold=threshold_scre,
                    layer=j,
                    train_threshold=train_threshold,
                )
                for j in range(depths)
            ]
        )
        
        self.dropout = nn.Dropout(0.5)
        self.classify = nn.Linear(self.embed_dims, num_classes, bias=0)

        self.spatial_pooling = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=[self.H_temp, self.W_temp], stride=[self.H_temp, self.W_temp], groups=self.embed_dims, bias=0)

        torch.manual_seed(42)
        nn.init.xavier_normal_(self.classify.weight)
        nn.init.xavier_normal_(self.spatial_pooling.weight)

        self.lif_head = LIFNeuron(threshold=threshold_head, decay=1, min=0.0, reset=True, name="lif_head", train=train_threshold)
        self.lif_pooling = LIFNeuron(threshold=threshold_head, decay=0, min=4/128, reset=True, name="lif_y_1", train=True)
        self.lif_result = LIFNeuron(threshold=threshold_head, decay=0, min=0.0, reset=True, name="lif_result", train=train_threshold)

    def _reset_neurons_(self, m):
        if isinstance(m, LIFNeuron):
            m.reset_state()
    
    def forward(self, x):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()
        T, B, C, H, W = x.shape
        self.apply(self._reset_neurons_)
        result_sum = torch.zeros(B, self.num_classes).to(device)
        for this_t in range(T):
            if self.dataset == "DVS128":
                threshold = 3
            else:
                threshold = 1 - (this_t)/T
          
            conv_out = self.conv(x[this_t].reshape(B, C, H, W).to(device), threshold=threshold)
            for _, blk in enumerate(self.blocks):
                screener_out = blk(conv_out)
            screener_out = self.lif_head(screener_out)

            head_input = screener_out
            pooling_out = self.spatial_pooling(head_input).reshape(B, self.embed_dims)
            classify_in = self.lif_pooling(pooling_out)
            result = self.classify(classify_in)
            result = self.lif_result(result)
            result_sum[:, :] = result_sum[:, :] + result 
            
        return result_sum
    
@register_model
def SV(**kwargs):
    model = SpikeVision(
     **kwargs
    )
    return model