import os
import unittest

import torch

from lipschitz.data.datasets.cifar10 import CIFAR10


class TestCIFAR10(unittest.TestCase):
    def test_evaluation_set_is_deterministic(self):
        ds = CIFAR10()
        train_eval1 = ds.get_data(use_test_data=False)
        train_eval2 = ds.get_data(use_test_data=False)

        e1, _ = train_eval1.eval[0]
        e2, _ = train_eval2.eval[0]

        self.assertTrue(torch.allclose(e1, e2))
