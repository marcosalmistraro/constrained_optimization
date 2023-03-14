# Script to implement re-usable OptNet layer
# Extension of the vanilla OptNet layer to the following problem formulation
#
# 1 - Non-Convex (NC) quadratic objective function
# 2 - Non-convex (NC) quadratic constraints
#
# Employing Sequential Quadratic Programming (SLSQP) to impose a series of 
# sequential quadratic assumptions and relax the problem. 
# According to the standard PyTorch implementation, we solve the forward pass
# and implicitly compute gradient adjustment during the backward oneV.

import torch
from torch import matmul, transpose
import torch.nn as nn
from torch.autograd import Variable

# Allocate to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Memory allocation to {device}")

class SLSQP(nn.Module):

    def __init__(self, x, U, f):
        super().__init__()
        self.x = Variable(x.to(device))
        self.U = nn.parameter(U.to(device))

        # Calculating q as gradient of the objective function w.r.t. current feasible solution
        # Since we assume that the current solution is feasible, the KKT conditions for the local
        # quadratic sub-problem are satisfied
        q = torch.tensor(requires_grad=True)
        f.backward()
        q = q.grad

