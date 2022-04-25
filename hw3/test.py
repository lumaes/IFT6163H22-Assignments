from torchviz import make_dot
import torch

a = torch.ones(2, requires_grad=True)
c = torch.ones(2, requires_grad=True)

b = (a**2 + c).sum()
make_dot(b)
