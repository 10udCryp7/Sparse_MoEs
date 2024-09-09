import sys
sys.path.insert(0, 'model')


import torch
import unittest
from head import *

class TestHead(unittest.TestCase):
    def test_head_init(self):
        head = Head(head_size = 16, n_embedding = 32)
        self.assertEquals(head.head_size, 16)
        self.assertEquals(head.n_embedding, 32)
    def test_head_forward(self):
        B,T,C = 4,8,32
        H = 16
        x = torch.randn(B,T,C)
        head = Head(head_size = H, n_embedding = C)
        k = head.key(x) # (B,T,C) @ (B,C,H) = (B,T,H)
        q = head.query(x) # (B,T,C) @ (B,C,H) = (B,T,H)
        v = head.value(x) # (B,T,C) @ (B,C,H) = (B,T,H)
        weight = q @ k.transpose(1,2)*H**(-0.5) # (B,T,H) @ (B,H,T) = (B,T,T)
        tril = torch.tril(torch.ones(T,T))
        weight = weight.masked_fill_(tril == 0, 0)
        weight = head.softmax(weight, -1)
        self.assertEquals(C, head.n_embedding)
        self.assertEquals(k.shape, (H,head.head_size))
        self.assertEquals(q.shape, (H,head.head_size))
        self.assertEquals(v.shape, (H,head.head_size))
        self.assertEquals(k.transpose(1,2).shape, (head.head_size,x[1]))
        self.assertEquals(weight.shape, (T,T))
        self.assertEquals(weight,torch.tril(weight))
        # test sum = 1
        for batch in weight:
            for row in batch:
                self.assertEquals(sum(row),1)

        weight = weight.masked_fill_(tril == 0, float('-inf'))
        out = weight @ v #(B,T,T) @ (B,T,H) = (B,T,H)
        self.assertEquals(out.shape,(B,T,H))

