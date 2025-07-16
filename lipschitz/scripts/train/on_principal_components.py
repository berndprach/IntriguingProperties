import argparse
import sys
import time
from functools import partial
from pathlib import Path

import torch

from lipschitz import data
from lipschitz.algorithms.pca import get_data_matrix, PCA
from lipschitz.data import TrainEvalDataLoader
from lipschitz.data.typing import TrainEval, TorchDataset
from lipschitz.io_functions import save_results, state_dict
from lipschitz.io_functions.parser import dictionary_str, iterable_str
from lipschitz.training.evaluate import evaluate
from lipschitz.training.trainer import Trainer
from lipschitz.scripts.train.configurations import DEFAULT_CONFIGURATIONS

parser = argparse.ArgumentParser()

add_arg = parser.add_argument
add_arg("-c", "--configuration-name", default="lipschitz_network",
        choices=DEFAULT_CONFIGURATIONS)
add_arg("-u", "--updates", type=dictionary_str, default="{}")
# add_arg("-pc", "--principal-components", type=iterable_str, default="[0]")
add_arg("-pc", "--principal-components", type=str, default="[0]")
add_arg("--save-as", default=None)
add_arg("-exp", "--experiment-name", default=None)  # Used for loading results.
add_arg("-s", "--split", default=None)  # Only used for loading results.


def main():
    config_train()


def config_train():
    a = parser.parse_args()
    print(f"Arguments: {a}")

    c = DEFAULT_CONFIGURATIONS[a.configuration_name].copy()
    c.update(*a.updates.items())
    print(c)

    # dl = data.get_loader(c["dataset"], c["preprocessing"])
    data_kwargs = {k: v for k, v in c["dataset"].items() if k != "batch_size"}
    d = data.get_data(**data_kwargs)

    assert c["preprocessing"]["name"] == "center"
    mean = torch.tensor(c["preprocessing"]["channel_means"])
    d = TrainEval(
        [(x - mean[:, None, None], y) for x, y in d.train],
        [(x - mean[:, None, None], y) for x, y in d.eval],
    )

    data_matrix = get_data_matrix(d.train)

    pca = PCA()
    pca.fit(data_matrix)

    components = iterable_str(a.principal_components)
    transform = partial(pca.reconstruct, components=components)

    d = TrainEval(
        [(transform(x), y) for x, y in d.train],
        [(transform(x), y) for x, y in d.eval],
    )

    dl = TrainEvalDataLoader(d, batch_size=c["dataset"]["batch_size"])
    dl.to_device()

    t = Trainer.from_config(c, total_steps=c["epochs"] * len(dl.train))
    t.full_training_run(c["epochs"], dl.train)

    output = {
        "command": " ".join(sys.argv),
        "arguments": vars(a),
        "configuration": c.as_dict,
        "filename": Path(__file__).name,
        "end_time": time.time(),
        "results": evaluate(t.model, dl, **c["evaluation"]),
    }
    fp = save_results(output)

    if a.save_as is not None:
        state_dict.save(a.save_as, t.model.state_dict(), fp)


if __name__ == "__main__":
    config_train()
