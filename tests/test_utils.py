import unittest

import torch
from model.utils import euclidean_dist

class UtilTest(unittest.TestCase):
    def test_euclidean_distance(self):
        a = torch.Tensor( [ [1,2], [3,4], [5,6] ] )
        b = torch.Tensor([[1,1],[1,1]])
        dists = euclidean_dist(a,b)
        print(dists)
        self.assertEqual(dists.shape, (3,2))  # add assertion here


if __name__ == '__main__':
    unittest.main()
