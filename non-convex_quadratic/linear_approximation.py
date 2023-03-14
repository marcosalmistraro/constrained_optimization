# Script to implement re-usable OptNet layer
# Extension of the vanilla OptNet layer to the following problem formulation
#
# 1 - Convex quadratic objective function
# 2 - Non-convex (NC) quadratic constraints
#
# Approximating equality constraints as a linear function employing first-order Taylor expansion
# NCL = (N)on-(C)onvex (L)inear approximation

from qpth.qp import QPFunction
import torch
from torch import matmul, transpose
import torch.nn as nn
from torch.autograd import Variable

# Allocate to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Memory allocation to {device}")

class NCLOptNetLayer(nn.Module):
    def __init__(self, x, U):
        self.x = torch.autograd.Variable(x.to(device))
        self.U = nn.parameter(U.to(device))

        # Approximate NC quadratic constraints and reduce to linear system in the form Ax = b
        self.A = nn.parameter(2 * (matmul(transpose(self.x), matmul(transpose(self.U), self.U))))
        self.b = nn.parameter(matmul(transpose(x, (matmul(matmul(transpose(self.U), self.U)), x))))

        # Solve linear system by computing inverse
        self.x = matmul(torch.inverse(self.A), self.b)
        return x        