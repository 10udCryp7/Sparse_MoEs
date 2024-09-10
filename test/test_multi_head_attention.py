import unittest
import torch
import sys
sys.path.insert(0, 'model')

from multi_head_attention import MultiHeadAttention
from head_attention import Head

class TestMultiHeadAttention(unittest.TestCase):
    def test_multi_head_init(self):
        mht = MultiHeadAttention(n_heads=4, head_size=16, embedding_size = 32)
        # check initialization and head_list
        self.assertEqual(len(mht.head_list), 4)
        self.assertEqual(mht.n_heads, 4)
        for i in range(4):
            self.assertTrue(isinstance(mht.head_list[i], Head))
    def test_multi_head_forward(self):
        mht = MultiHeadAttention(n_heads=4, head_size=16, embedding_size = 32)
        B, T, C = 4, 8, 32
        x = torch.randn(B, T, C)
        # check concat
        concat_out = mht.concat(x)  # (B,T,n_head*H)
        self.assertEqual(concat_out.shape, torch.Size(
            [B, T, mht.n_heads*mht.head_size]))
        # check fully connected
        fc_out = mht.fc(concat_out)  # (B,T,head_size)
        self.assertEqual(fc_out.shape, torch.Size([B, T, mht.head_size]))

        # check output
        out = mht.dropout(fc_out)
        self.assertEqual(out.shape, torch.Size([B, T, mht.head_size]))
        self.assertEqual(out.shape, mht.forward(x).shape)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)