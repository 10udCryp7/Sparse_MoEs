import torch
from torch import nn

from head_attention import *
# class MultiHeadAttention(nn.Module):
#     def __init__(self, n_heads):
#         super(MultiHeadAttention,self).__init__()
#         M


x = torch.randn(4,8,32)
mht = nn.ModuleList([Head(16,32) for _ in range(4)])

print(type(mht[0]))