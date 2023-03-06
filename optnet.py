import torch

import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from qpth.qp import QPFunction, QPSolvers

class Optnet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn:bool, nineq=200, neq=0, eps=1e-4):
        super().__init__()

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls

        # Possibly apply batch normalization according to Ioffe & Szegedy (2015).
        # This allows to maintain higher learning rates while being less careful
        # about initialization
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)
        
        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)
        
        # Insert debugging tool for the value of neq 
        assert(neq==0), "parameter neq should be set to 0"

        # Create layer variables and push them to GPU by employing .cuda()
        self.M = Variable(torch.tril(torch.ones(nCls, nCls)).cuda())
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls)).cuda())
        self.G = Parameter(torch.Tensor(nineq, nCls).uniform_(-1, 1).cuda())
        self.z0 = Parameter(torch.zeros(nCls).cuda())
        self.s0 = Parameter(torch.ones(nineq).cuda())

        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        def forward(self, x):

            # Select input's row as batch size for the forward pass
            nBatch = x.size([0])

            # Implement the model's structure as:
            # - Fully Connected layer
            # - ReLu
            # - Batch Normalization
            # - Fully Connected layer
            # - ReLu
            # - Batch Normalization
            # - Quadratic Problem
            # - SoftMax (to ensure readability in the context of a classification problem)
        
            # Rescale x tensor to have nBatch rows
            x = x.view(nBatch, -1)

            # Implement first fully-connected layer
            x = F.relu(self.fc1(x))
            # Possibly apply batch normalization
            if self.bn:
                x = self.bn1(x)
            # Implement second fully-connected layer
            x = F.relu(self.fc2(x))
            # Possibly apply batch normalization
            if self.bn:
                x = self.bn2(x)
            
            #TODO implement final QP problem


            # Apply final activation function
            return F.log_softmax(x)