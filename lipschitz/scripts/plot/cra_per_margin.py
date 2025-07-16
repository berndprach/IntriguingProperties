import argparse
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from io_functions import state_dict

# OUTPUT_FOLDER = Path("outputs")
MARGIN_DIR = Path("outputs", "margins")
MARGIN_NAMES = [f for f in os.listdir(MARGIN_DIR) if f.endswith(".yaml")]

arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("-d", "--dataset", default="CIFAR10")
# arg_parser.add_argument("-s", "--state-dict-name", optional=state_dict.NAMES)
arg_parser.add_argument("-m", "--margin-file-name", choices=MARGIN_NAMES)
arg_parser.add_argument("-c", "--correction", type=float, default=2**0.5)
arg_parser.add_argument("--exclude-train", action="store_true")
arg_parser.add_argument("--exclude-eval", action="store_true")


plt.rcParams.update({'font.size': 14})

CRA = "Certified Robust Accuracy"


def plot_cra_per_margin(*args):
    """
    First run:
    >> python .\lipschitz\run.py evaluate_margins -s STATE-DICT-NAME.pt
    """

    arg = arg_parser.parse_args(args)
    print(f"Arguments: {arg}")

    margin_path = os.path.join(MARGIN_DIR, arg.margin_file_name)
    with open(margin_path, "r") as f:
        margins = yaml.safe_load(f)
    print(f"Loaded margins from {margin_path}.")

    drp = partial(draw_robustness_plot, correction=arg.correction)

    if not arg.exclude_train:
        drp(margins["train"], label="train")

    if not arg.exclude_eval:
        drp(margins["eval"], label="evaluation")

    plt.legend()

    # plt.title(f"CRA for {MODEL_NAME.get(arg.setting, arg.setting)}")
    format_plot()

    # fn = f"s{arg.setting:02d}_{arg.epochs:03d}ep_id{arg.offset_id}.pdf"
    # if arg.test:
    #     fn = "test_" + fn
    # plot_folder = OUTPUT_FOLDER / arg.folder_name / "plots"
    # plot_folder.mkdir(parents=True, exist_ok=True)
    # plot.savefig(plot_folder / fn)

    plt.show()





def draw_stars(results, partition="val"):
    xs = [0., 36 / 255, 72 / 255, 108 / 255, 1.]
    names = ["Accuracy", "CRA0.20", "CRA0.40", "CRA0.60", "CRA1.41"]
    val_results = [results[partition][name] * 100 for name in names]
    plt.scatter(xs, val_results, marker="*", color="orange")
    for x, y in zip(xs, val_results):
        plt.text(x, y, f"{y:.1f}%", verticalalignment="bottom")


def format_plot():
    # plt.title("Certified Robustness")
    # plt.title(CRA)
    plt.xlabel("Perturbation size")
    # plt.ylabel("Accuracy (%)")
    plt.ylabel(f"{CRA} (%)")

    # plt.xlim(left=-.01, right=2.01)
    plt.xlim(left=-0., right=2.)
    # plt.xlim(left=-0., right=1.2)
    plt.ylim(0, 100)

    # y_ticks = [10*i for i in range(11)]
    # plt.yticks(y_ticks, [f"{y}" for y in y_ticks])
    # major_y_ticks = [0, 50, 100]
    major_y_ticks = [20, 40, 60, 80, 100]
    minor_y_ticks = [10*i for i in range(1, 10)]
    plt.yticks(major_y_ticks, [f"{y}" for y in major_y_ticks])
    plt.yticks(minor_y_ticks, ["" for _ in minor_y_ticks], minor=True)

    major_x_ticks = [0, 0.5, 1, 1.5, 2]
    plt.xticks(major_x_ticks, [f"{x:.1f}" for x in major_x_ticks])

    for p in [36/255, 72/255, 108/255]:
        plt.axvline(p, color="gray", linestyle="--", alpha=0.5)

    # major_x_ticks = [0, 36/255, 72/255, 108/255, 1]
    # x_tick_labels = ["0", r"$\frac{36}{255}$", r"$\frac{72}{255}$", r"$\frac{108}{255}$", "1"]
    # plt.xticks(major_x_ticks, x_tick_labels)

    plt.grid()
    plt.grid(which="minor")
    # plt.legend()


def draw_robustness_plot(margins, correction=2**0.5, **kwargs):
    margins.sort()
    acc_left = [1 - i/len(margins) for i in range(len(margins)+1)]

    radii = [m / correction for m in margins]
    xs = duplicate(radii)
    ys = [acc_left[0], *duplicate(acc_left[1:-1]), acc_left[-1]]
    ys = [100*y for y in ys]

    if xs[0] > 0.:
        xs = [0., *xs]
        ys = [100., *ys]

    plt.plot(xs, ys, **kwargs)
    plt.fill_between(xs, ys, alpha=0.2)


def duplicate(values: list):
    return [v for v in values for _ in (0, 1)]


def main(*args):
    plot_main_results(*args)
