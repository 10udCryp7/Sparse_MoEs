import torch
from torch import nn
from head_attention import Head


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, embedding_size):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.embedding_size = embedding_size
        self.head_list = [Head(
            head_size=self.head_size, embedding_size=self.embedding_size) for _ in range(n_heads)]
        self.concat = lambda x: torch.concat([head(x) for head in self.head_list], dim=-1)
        self.fc = nn.Linear(n_heads*head_size, head_size)
        self.dropout = nn.Dropout()

    def forward(self, x):
        concat_out = self.concat(x)
        fc_out = self.fc(concat_out)
        out = self.dropout(fc_out)
        return out  
