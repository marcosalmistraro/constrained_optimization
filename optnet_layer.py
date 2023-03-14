# Script to implement re-usable OptNet layer

from qpth.qp import QPFunction
import torch
import torch.nn as nn
from torch.autograd import Variable

# Allocate to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Memory allocation to {device}")

class OptNetLayer(nn.Module):
    def __init__(self, x, Q, G, h):
        self.x = torch.nn.parameter(x.to(device))
        self.Q = torch.nn.parameter(Q.to(device))
        self.G = torch.nn.parameter(G.to(device))
        self.h = torch.nn.parameter(h.to(device))
    
    def forward(self):
        e = Variable(torch.tensor())
        x = QPFunction(self.Q, self.x, self.G, self.h, e, e)
        return x