import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from lipschitz.io_functions.parser import dictionary_str
from lipschitz.io_functions.plotting import plot_line
from lipschitz.io_functions.result_parsing import get_results, filter_results

plt.rcParams.update({'font.size': 14})

# RESULTS_FOLDER = Path("outputs", "results")
PLOT_FOLDER = Path("outputs", "result_plots")
PLOT_FOLDER.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-c", "--constraints", type=dictionary_str, default="{'epochs': 100}")
add_arg("-s", "--split-by", type=str, default="arguments.model_name")
add_arg("-x", "--x-key", type=str, default="arguments.lr")
add_arg("-xs", "--x-scale", choices=["linear", "log"], default="linear")
add_arg("-y", "--y-key", type=str, default="results.eval.CRA(0.14)")
# add_arg("-ys", "--y-scale", choices=["linear", "log"], default="linear")
add_arg("-u", "--unique-values", action="store_true", default=False)


def main():
    plot_results()


def plot_results():
    args = parser.parse_args()
    print(f"Arguments: {args}")

    data, split_values = get_data(
        args.constraints, args.x_key, args.y_key, args.split_by
    )

    yd = {s: {args.x_key: xs, args.y_key: ys} for s, (xs, ys) in data.items()}
    print("\nData for plotting:")
    print(yaml.safe_dump(yd, default_flow_style=None))

    # for result in results:
    #     print(result["file_path"])

    for split_name, (xs, ys) in data.items():
        plt.scatter(xs, ys, label=split_name)
        plot_line(xs, ys)

        # max_xy = max(zip(xs, ys), key=lambda xy: xy[1])
        # print(f"Max for split {sv}: x={max_xy[0]}, y={max_xy[1]}")
        #
        # min_xy = min(zip(xs, ys), key=lambda xy: xy[1])
        # print(f"Min for split {sv}: x={min_xy[0]}, y={min_xy[1]}")

    plt.title(f"{args.y_key}")
    plt.xlabel(f"{args.x_key} ({args.x_scale})")
    plt.ylabel(args.y_key)

    plt.xscale(args.x_scale)

    legend_title = " ".join(args.split_by.split(".")[-2:])
    legend_title = legend_title.replace("_", " ").capitalize() + ":"
    if 0 < len(split_values) <= 10:
        plt.legend(title=legend_title)
    if 10 < len(split_values) < 20:
        plt.legend(title=legend_title,
                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
    plt.grid()

    plt.show()


def get_data(constraints, x_key, y_key, split_by):
    results = get_results()
    results = filter_results(results, constraints)
    results = filter_results(results, {}, [x_key, y_key])
    split_values = set(r.get(split_by) for r in results)
    print(f"\nSplitting by {split_by} with values {split_values}.")
    data = {}
    for sv in sorted(split_values, key=str):
        split_results = [r for r in results if r.get(split_by) == sv]
        xs = [r[x_key] for r in split_results]
        ys = [r[y_key] for r in split_results]

        data[sv] = (xs, ys)

    return data


if __name__ == "__main__":
    main()
