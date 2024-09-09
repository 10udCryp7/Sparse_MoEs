import sys
sys.path.insert(0, 'model')


import torch
import unittest
from head import *

class TestHead(unittest.TestCase):
    def test_head_init(self):
        head = Head(head_size = 16, n_embedding = 32)
        self.assertEqual(head.head_size, 16)
        self.assertEqual(head.n_embedding, 32)
    def test_head_forward(self):
        B,T,C = 4,8,32
        H = 16
        x = torch.randn(B,T,C)
        head = Head(head_size = H, n_embedding = C)
        tril = torch.tril(torch.ones(T,T))
        k = head.key(x) # (B,T,C) @ (B,C,H) = (B,T,H)
        q = head.query(x) # (B,T,C) @ (B,C,H) = (B,T,H)
        v = head.value(x) # (B,T,C) @ (B,C,H) = (B,T,H)
        weight = q @ k.transpose(1,2)*H**(-0.5) # (B,T,H) @ (B,H,T) = (B,T,T)
        weight = weight.masked_fill_(tril == 0, float('-inf'))
        weight = head.softmax(weight, -1)
        self.assertEqual(C, head.n_embedding)
        self.assertEqual(k.shape, torch.Size([B,T,H]))
        self.assertEqual(q.shape, torch.Size([B,T,H]))
        self.assertEqual(v.shape, torch.Size([B,T,H]))
        self.assertEqual(k.transpose(1,2).shape, torch.Size([B,H,T]))
        self.assertEqual(weight.shape, torch.Size([B,T,T]))
        self.assertEqual(weight,torch.tril(weight))
        # test sum = 1
        for batch in weight:
            for row in batch:
                self.assertAlmostEqual(torch.sum(row).item(), 1)
        weight = head.dropout(weight)
        out = weight @ v #(B,T,T) @ (B,T,H) = (B,T,H)
        self.assertEqual(out.shape,torch.Size([B,T,H]))

