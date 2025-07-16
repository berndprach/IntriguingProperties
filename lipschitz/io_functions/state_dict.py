import os
from pathlib import Path

import torch
import yaml
from torch import nn

from lipschitz import models

STATE_DICT_FOLDER = Path("outputs", "state_dicts")
os.makedirs(STATE_DICT_FOLDER, exist_ok=True)
MAPPING_FILE = STATE_DICT_FOLDER / "name_to_result.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NAMES = [n for n in sorted(os.listdir(STATE_DICT_FOLDER)) if n.endswith(".pt")]


def get_mapping() -> dict[str, str]:
    with open(MAPPING_FILE, "r") as f:
        lines = f.readlines()
    mapping = {}
    for line in lines:
        name, path = line.strip().split(": ")
        if not name.endswith(".pt"):
            name += ".pt"
        mapping[name] = path
    return mapping


def save(name: str, state_dict: dict, results_path: str = None):
    # Get state-dict as >> state_dict = model.state_dict()
    filename = f"{name}.pt"
    fp = STATE_DICT_FOLDER / filename

    torch.save(state_dict, fp)
    print(f"\nSaved model state-dict to {fp}.")

    with open(MAPPING_FILE, "a") as f:
        f.write(f"{filename}: {results_path}\n")


def load_weights(model: nn.Sequential, state_dict_name: str) -> None:
    sd_path = STATE_DICT_FOLDER / state_dict_name
    print(f"\nLoading model state-dict from {sd_path}...")
    state_dict = torch.load(sd_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)


# load_weight = partial(
#     torch.load, map_location=torch.device('cpu'), weights_only=True
# )


def get_model_kwargs(state_dict_name: str) -> dict:
    results = load_results(state_dict_name)
    return results["configuration.model"]


def load_results(state_dict_name: str) -> dict:
    result_path_from_name = get_mapping()
    results_fn = result_path_from_name[state_dict_name]
    with open(results_fn, "r") as f:
        results = yaml.safe_load(f)
    return results


def load_trained_model(state_dict_name: str) -> nn.Module:
    mkw = get_model_kwargs(state_dict_name)
    model = models.load(**mkw)
    load_weights(model, state_dict_name)
    return model
