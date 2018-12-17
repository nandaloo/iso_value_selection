import unittest
import numpy as np

from utils import normalize_to_indexes


class TestIndexNormalization(unittest.TestCase):

    def test_normalize_1d(self):

        self.assertTrue(np.allclose(
            normalize_to_indexes(n=3, d=1),
            [[0, 0.5, 1]]
        ))

        self.assertTrue(np.allclose(
            normalize_to_indexes(x=[1, 2, 3]),
            [[1,2,3]]
        ))

        self.assertTrue(np.allclose(
            normalize_to_indexes(x=[1, 2, 3]),
            [[1,2,3]]
        ))

        self.assertTrue(np.allclose(
            normalize_to_indexes(data=[[0, 10]], d=1, n=3),
            [0, 5, 10]
        ))

        self.assertTrue(np.allclose(
            normalize_to_indexes(data=[[0, 10]], d=1, n=3),
            [[0, 5, 10]]
        ))

    def test_normalize_2d(self):

        self.assertTrue(np.allclose(
            normalize_to_indexes(shape=(3, 3)),
            [[0, 0.5, 1]]*2
        ))

        self.assertTrue(np.allclose(
            normalize_to_indexes(n=3, d=2),
            [[0, 0.5, 1]]*2
        ))

        self.assertTrue(np.allclose(
            normalize_to_indexes(x=[1, 2, 3], y=[2,3,4]),
            [[1,2,3], [2,3,4]]
        ))

        self.assertTrue(np.allclose(
            normalize_to_indexes(data=[[0, 10], [10, 20]], n=3),
            [[0, 5, 10], [10, 15, 20]]
        ))


if __name__ == '__main__':
    unittest.main()