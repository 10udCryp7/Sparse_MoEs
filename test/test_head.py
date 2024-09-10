from head import *
import unittest
import torch
import sys
sys.path.insert(0, 'model')


class TestHead(unittest.TestCase):
    def test_head_init(self):
        head = Head(head_size=16, n_embedding=32)
        self.assertEqual(head.head_size, 16)
        self.assertEqual(head.n_embedding, 32)

    def test_head_forward(self):
        B, T, C = 4, 8, 32
        H = 16
        x = torch.randn(B, T, C)
        head = Head(head_size=H, n_embedding=C)
        tril = torch.tril(torch.ones(T, T))
        q = head.query(x)  # (B,T,C) @ (B,C,H) = (B,T,H)
        k = head.key(x)  # (B,T,C) @ (B,C,H) = (B,T,H)
        v = head.value(x)  # (B,T,C) @ (B,C,H) = (B,T,H)
        weight = q @ k.transpose(1, 2)*H**(-0.5)  # (B,T,H) @ (B,H,T) = (B,T,T)
        weight = weight.masked_fill(tril == 0, 0)
        self.assertEqual(C, head.n_embedding)
        self.assertEqual(q.shape, torch.Size([B, T, H]))
        self.assertEqual(k.shape, torch.Size([B, T, H]))
        self.assertEqual(v.shape, torch.Size([B, T, H]))
        self.assertEqual(k.transpose(1, 2).shape, torch.Size([B, H, T]))
        self.assertEqual(weight.shape, torch.Size([B, T, T]))
        print(f'{weight =}')
        print(f'{head.tril(weight) =}')
        self.assertTrue(torch.equal(weight, head.tril(weight)))

        weight = weight.masked_fill(tril == 0, float('-inf'))
        weight = head.softmax(weight)
        # test sum = 1
        for batch in weight:
            for row in batch:
                self.assertAlmostEqual(torch.sum(row).item(), 1.0, delta=1e4)

        weight = head.dropout(weight)
        out = weight @ v  # (B,T,T) @ (B,T,H) = (B,T,H)
        self.assertEqual(out.shape, torch.Size([B, T, H]))
        self.assertEqual(out.shape, head.forward(x).shape)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
