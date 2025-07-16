from math import sqrt

from lipschitz.data.datasets import cifar10
from lipschitz.io_functions.configuration import Configuration

DEFAULT_CONFIGURATION_YAML = f"""
epochs: 24
dataset:
    name: CIFAR10
    use_test_data: false
    training_size: null
    batch_size: 256
model:
    name: SimpleConvNet
augmentation:
    name: 94percent
loss:
    name: OffsetCrossEntropy
    offset: 0.
    temperature: 8.
optimizer:
    name: SGD
    lr: 0.1
    momentum: 0.9
    nesterov: true
scheduler:
    name: OneCycleLR
evaluation:
    name: Accuracy
    partitions: ["train", "eval"]
preprocessing:
    name: center
    channel_means: {list(cifar10.CHANNEL_MEANS)}
"""

DEFAULT_CONFIGURATION = Configuration.from_yaml(DEFAULT_CONFIGURATION_YAML)
