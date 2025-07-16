# IntriguingProperties

This repository contains code to reproduce the experiments from our paper 
[Intriguing Properties of Robust Classification](https://arxiv.org/pdf/2412.04245).
It includes our e.g. our scripts for 
[training](lipschitz/scripts/train/main.py) models and 
[plotting](lipschitz/scripts/plot/results.py) results.

## Usage:

### Prerequisites:
```bash
python -m venv venv
source venv/bin/activate
python -m pip install .
```

### Train models:
```bash
train -c="simple_conv_net"
train -c="lipschitz_network" -u="{'model.name': 'AOL-MLP'}"
train -c="lipschitz_network" -u="{'model.name': 'CPL-LCN', 'optimizer.lr': 1.}"
train -c="randomized_smoothing"
```

## Reproduce Experiments from the Paper:
Follow the following pseudocode to run experiments similar to the ones in the paper:

### Learning rate search:
```bash
for _ in range(16):
    train -exp="lr-search" -u="{'epochs': 10, 'optimizer.lr': 10**random.uniform(-3, 1)}"
    train -exp="lr-search" -u="{'model.name': 'CPL-LCN', 'epochs': 10, 'optimizer.lr': 10**random.uniform(-3, 1)}"
    train -exp="lr-search" -c="randomized_smoothing" -u="{'epochs': 10, 'optimizer.lr': 10**random.uniform(-3, 1)}" 
plot -c="{'arguments.experiment_name': 'lr-search'}" -s="configuration.model.name" -x="configuration.optimizer.lr" -xs=log
```

### Robust scaling behavior:
```bash
for TRAINING_SIZE in [48, 97, 195, 390, 781, 1_562, 3_125, 6_250, 12_500, 25_000, 50_000]:
    train -exp="scaling" -u="{'epochs': 3000*50000//TRAINING_SIZE, 'dataset.training_size': TRAINING_SIZE, 'optimizer.lr': 0.1}"
    train -exp="scaling" -u="{'model.name': 'CPL-LCN', 'epochs': 3000*50000//TRAINING_SIZE, 'dataset.training_size': TRAINING_SIZE, 'optimizer.lr': 1.}"
    train -exp="scaling" -c="randomized_smoothing" -u="{'epochs': 3000*50000//TRAINING_SIZE, 'dataset.training_size': TRAINING_SIZE, 'optimizer.lr': 0.3}"
plot -c="{'arguments.experiment_name': 'scaling'}" -s="configuration.model.name" -x="configuration.dataset.training_size" -xs="log"
```

### Scaling behavior with EDM data:
```bash
for TRAINING_SIZE in [30, 61, 122, 244, 488, 976, 1_953, 3_906, 7_812, 15_625, 31_250, 62_500, 125_000, 250_000, 500_000, 1_000_000]:
    train -exp="edm_scaling" -u="{'epochs': 3_000, 'dataset.name': 'EDMCIFAR10', 'dataset.training_size': TRAINING_SIZE, 'optimizer.lr': 0.1}"
    train -exp="edm_scaling" -u="{'model.name': 'CPL-LCN', 'epochs': 3_000, 'dataset.name': 'EDMCIFAR10', 'dataset.training_size': TRAINING_SIZE, 'optimizer.lr': 1.}"
    train -exp="edm_scaling" -c="randomized_smoothing" -u="{'epochs': 100, 'dataset.name': 'EDMCIFAR10', 'dataset.training_size': TRAINING_SIZE, 'optimizer.lr': 0.3}"
plot -c="{'arguments.experiment_name': 'edm_scaling'}" -s="configuration.model.name" -x="configuration.dataset.training_size" -xs="log"
```


### Robust and non-robust features:
```bash
for COMPONENTS in ["range(16)", "range(512)", "range(512, 3072)", "range(2048, 3072)", "list(range(16)) + list(range(512, 3072))"]:
    python lipschitz/scripts/train/on_principal_components.py -exp="pc" -c="simple_conv_net" -pc=COMPONENTS -u="{'epochs': 300}"
python lipschitz/scripts/plot/result_bars.py -c="{'arguments.experiment_name': 'pc'}" -s="configuration.model.name" -x="arguments.principal_components" -y="results.eval.Accuracy()"

for COMPONENTS in ["range(16)", "range(512)", "range(512, 3072)", "range(2048, 3072)", "list(range(16)) + list(range(512, 3072))"]:
    python lipschitz/scripts/train/on_principal_components.py -exp="pc" -pc=COMPONENTS -u="{'epochs': 3_000}"
python lipschitz/scripts/plot/result_bars.py -c="{'arguments.experiment_name': 'pc'}" -s="configuration.model.name" -x="arguments.principal_components"
```
