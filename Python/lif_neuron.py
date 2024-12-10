import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import gradcheck

device = torch.device("cuda:0") if torch.cuda.is_available()==True else torch.device('cpu') 


def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

def sigmiod(x):
    return 2 / (1 + np.exp(-x)) - 2 / (1 + np.exp(-x+1))

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.ge(0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 15.0
        hight = .15
        lens = .1
        mul = 0.9
        gamma = 1

        
        temp = torch.sigmoid(gamma * input) * (1 - torch.sigmoid(gamma * input))

        return grad_input * temp.float() * 3 # stablize the training


act_fun = ActFun.apply
    
class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=1, min=8/128, reset=True, name=None, train=True):
        pass
        super(LIFNeuron, self).__init__()
        self.decay = decay
        if train:
            self.threshold = nn.Parameter(torch.tensor([threshold], dtype=torch.float))
        else:
            self.threshold = torch.tensor([threshold], dtype=torch.float).to(device)
        self.reset = reset
        self.mem = None
        self.name = name
        self.min = min

    def forward(self, inputs, threshold=None):
        if self.mem is None:
            self.mem = torch.zeros_like(inputs)
        
        self.threshold.data.clamp_(min=self.min)
        self.mem = self.decay * self.mem + inputs
        if threshold is None:
            spike = act_fun(0.4 * torch.tanh(self.mem - self.threshold))
        else:
            spike = act_fun(0.4 * torch.tanh(self.mem - threshold))
        

        if self.reset:
            self.mem *= (1 - spike)
        else:
            self.mem -= self.threshold * spike
        
        negative_ = (self.mem < 0).float()
        self.mem *= (1 - negative_)

        return spike
    
    def reset_state(self):
        self.mem = None 

    def threshold_positive(self):
        with torch.no_grad():
            self.threshold.data.clamp_(min=0.5)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, threshold={self.threshold.item()})"
    
if __name__ == "__main__":
    input = torch.randn((1, 1), dtype=torch.double, requires_grad=True)
    test = gradcheck(act_fun, input, eps=1e-6, atol=1e-4)

    print("Gradcheck passed:", test)