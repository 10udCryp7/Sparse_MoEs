import torch
from torch import nn


class Expert(nn.Module):
    def __init__(self, head_size):
        self.head_size = head_size
        super(Expert, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(head_size, head_size*4),
            nn.ReLU(),
            nn.Linear(4*head_size, head_size),
            nn.Dropout()
        )

    def forward(self, x):
        return self.expert(x)
    



