import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt

from io_functions.parser import dictionary_str
from io_functions.plotting import plot_line, nice_ticks
from io_functions.result_parsing import (
    get_results, get_item, get, similar, filter_results
)

plt.rcParams.update({'font.size': 14})


RESULTS_FOLDER = Path("outputs", "results")
PLOT_FOLDER = Path("outputs", "result_plots")
PLOT_FOLDER.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-c", "--constraints", type=dictionary_str, default="{'epochs': 100}")
add_arg("-s", "--split-by", type=str, default="arguments.model_name")
add_arg("-x", "--x-key", type=str, default="arguments.lr")
add_arg("-xs", "--x-scale", choices=["linear", "log"], default="linear")
add_arg("-y", "--y-key", type=str, default="results.eval.CRA(0.14)")
add_arg("-ys", "--y-scale", choices=["linear", "log"], default="linear")
add_arg("-g", "--goal", type=str, default=None)
add_arg("--min", action="store_true", default=False)


def plot_arg_max(*args):
    args = parser.parse_args(args)
    print(f"Arguments: {args}")

    if args.goal is None:
        args.goal = args.y_key

    _best = min if args.min else max
    best = partial(_best, key=lambda r: get_item(args.goal, r))

    results = get_results()
    results = filter_results(results, args.constraints)
    results = filter_results(results, {}, [args.x_key, args.y_key, args.goal])

    split_values = set(get(args.split_by, r) for r in results)
    print(f"Splitting by {args.split_by} with values {split_values}.")

    # Get data:
    arg_max_run = {}
    for sv in sorted(split_values, key=str):
        arg_max_run[sv] = {}
        split_results = [r for r in results if get(args.split_by, r) == sv]
        xs = set(get_item(args.x_key, r) for r in split_results)
        for x in xs:
            x_results = [r for r in split_results
                         if similar(get_item(args.x_key, r), x)]
            print(f"s={sv}, {args.x_key}={x}: {len(x_results)} results.")
            arg_max_run[sv][x] = best(x_results)

    # Plot data:
    for split_value, arg_max_runs in arg_max_run.items():
        xs = list(arg_max_runs.keys())
        ys = [get_item(args.y_key, r) for r in arg_max_runs.values()]

        plt.scatter(xs, ys, label=split_value)
        plot_line(xs, ys)
        d = {x: round(y, 3) for x, y in zip(xs, ys)}
        print(f"\"{split_value}-{args.y_key}\": {d}")

    plt.title(f"{args.y_key}")
    plt.xlabel(f"{args.x_key} ({args.x_scale})")
    plt.ylabel(args.y_key)

    plt.xscale(args.x_scale)
    plt.yscale(args.y_scale)

    all_runs = [r for runs in arg_max_run.values() for r in runs.values()]

    x_values = [get_item(args.x_key, r) for r in all_runs]
    nice_ticks(x_values, plt.xticks)

    y_values = [get_item(args.y_key, r) for r in all_runs]
    nice_ticks(y_values, plt.yticks)

    legend_title = args.split_by.replace("_", " ").capitalize() + ":"
    if 0 < len(split_values) <= 10:
        plt.legend(title=legend_title)
    if 10 < len(split_values) < 20:
        plt.legend(title=legend_title,
                   bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize='small')
        plt.tight_layout()
    plt.grid()

    plt.show()


