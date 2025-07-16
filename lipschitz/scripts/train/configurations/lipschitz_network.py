from math import sqrt

from lipschitz.data.datasets import cifar10
from lipschitz.io_functions.configuration import Configuration

DEFAULT_CONFIGURATION_YAML = f"""
epochs: 100
dataset:
    name: CIFAR10
    use_test_data: false
    training_size: null
    batch_size: 256
model:
    name: AOL-MLP
augmentation:
    name: 94percent
loss:
    name: OffsetCrossEntropy
    offset: {sqrt(2) * 36 / 255}
    temperature: {1 / 4}
optimizer:
    name: SGD
    lr: 0.1
    momentum: 0.9
    nesterov: true
scheduler:
    name: OneCycleLR
evaluation:
    name: Lipschitz
    partitions: ["train", "eval"]
preprocessing:
    name: center
    channel_means: {list(cifar10.CHANNEL_MEANS)}
"""

DEFAULT_CONFIGURATION = Configuration.from_yaml(DEFAULT_CONFIGURATION_YAML)
