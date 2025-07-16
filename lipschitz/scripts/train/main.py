import argparse
import sys
import time
from pathlib import Path

from lipschitz import data
from lipschitz.io_functions import save_results, state_dict
from lipschitz.io_functions.parser import dictionary_str
from lipschitz.training.evaluate import evaluate
from lipschitz.training.trainer import Trainer
from lipschitz.scripts.train.configurations import DEFAULT_CONFIGURATIONS

DC = list(DEFAULT_CONFIGURATIONS.keys())

parser = argparse.ArgumentParser()

add_arg = parser.add_argument
add_arg("-c", "--configuration-name", default="lipschitz_network", choices=DC)
add_arg("-u", "--updates", type=dictionary_str, default="{}")
add_arg("--save-as", default=None)
add_arg("-exp", "--experiment-name", default=None)  # Purely for result filter.
add_arg("-s", "--split", default=None)  # Purely for results separating.


def main():
    a = parser.parse_args()
    print(f"Arguments: {a}")

    config_train(a)


def config_train(a):
    c = DEFAULT_CONFIGURATIONS[a.configuration_name].copy()
    c.update(*a.updates.items())
    print(c)

    dl = data.get_loader(c["dataset"], c["preprocessing"])

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
    main()
