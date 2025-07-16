import os
from datetime import datetime
from pathlib import Path

import yaml

RESULTS_FOLDER = Path("outputs", "results")


def save_results(results: dict):
    fp = get_unique_filename()

    with open(fp, "w") as f:
        yaml.safe_dump(results, f)
    print(f"Saved results to {fp}.")

    print(f"Results:")
    print(yaml.safe_dump(results).replace("  ", "    "))

    return fp


def get_unique_filename():
    filename = datetime.now().strftime("%Y/%m/%d/%H_%M_I.yaml")
    fp = os.path.join(RESULTS_FOLDER, *filename.split("/"))
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    while os.path.exists(fp):
        fp = fp.replace(".yaml", "I.yaml")
    return fp
