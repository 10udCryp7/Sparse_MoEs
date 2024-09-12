import unittest
import torch
import sys
sys.path.insert(0, 'model')

from expert import Expert

class TestExpert(unittest.TestCase):
    def test_init(self):
        expert = Expert(20)
        self.assertEqual(expert.head_size, 20)
    def test_forward(self):
        expert = Expert(20)
        x = torch.randn(4,8,20)
        self.assertEqual(expert.forward(x).shape, torch.Size([4,8,20]))



if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)