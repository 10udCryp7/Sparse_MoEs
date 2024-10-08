import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, head_size, embedding_size):
        super(Head, self).__init__()
        # size of head
        self.head_size = head_size
        # size of vector embedding
        self.n_embedding = embedding_size
        self.query = nn.Linear(self.n_embedding, self.head_size, bias=False)
        self.key = nn.Linear(self.n_embedding, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embedding, self.head_size, bias=False)
        '''
        if we assign value for tril hear instead, 
        we need to use register_buffer as buffer wont be returned in model.parameters() so that optimizer wont update tril.
        '''
        self.tril = torch.tril
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        tril = self.tril(torch.ones(x.shape[1], x.shape[1]))
        weight = q @ k.transpose(1, 2)*self.head_size**(-0.5)
        weight = weight.masked_fill(tril == 0, float('-inf'))
        weight = self.softmax(weight)
        weight = self.dropout(weight)
        out = weight @ v
        return out
