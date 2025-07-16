import unittest

from lipschitz.io_functions import configuration


class TestConfiguration(unittest.TestCase):
    def test_single_update(self):
        c = {'learning_rate': 0.1, 'dataset': {'name': 'CIFAR10'}}

        configuration.nested_update(c, ("dataset.size", 10_000))

        self.assertEqual(c['dataset']['size'], 10_000)

    def test_multiple_updates(self):
        c = {'learning_rate': 0.1, 'dataset': {'name': 'CIFAR10'}}
        updates = {"dataset.size": 10_000, "learning_rate": 1.}

        configuration.nested_update(c, *updates.items())

        self.assertEqual(c['dataset']['size'], 10_000)
        self.assertEqual(c['learning_rate'], 1.)

    def test_update_with_dictionary(self):
        c = {'learning_rate': 0.1, 'dataset': {'name': 'CIFAR10'}}

        configuration.nested_update(c, ("dataset", {"source": "disk"}))

        self.assertTrue("name" not in c['dataset'].keys())
        self.assertEqual(c['dataset']['source'], "disk")
