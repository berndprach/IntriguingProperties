import argparse

import matplotlib.pyplot as plt
import yaml

from lipschitz.io_functions.parser import dictionary_str
from lipschitz.io_functions.plotting import plot_line, outside_legend
from lipschitz.io_functions.result_parsing import get_results, filter_results

plt.rcParams.update({'font.size': 14})

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-c", "--constraints", type=dictionary_str, default="{}")
add_arg("-s", "--split-by", type=str, default="arguments.model_name")
add_arg("-x", "--x-key", type=str, default="arguments.lr")
add_arg("-xs", "--x-scale", choices=["linear", "log"], default="linear")
add_arg("-y", "--y-key", type=str, default="results.eval.CRA(0.14)")


def main():
    plot_results()


def plot_results():
    args = parser.parse_args()
    print(f"Arguments: {args}")

    data = get_data(args.constraints, args.x_key, args.y_key, args.split_by)
    print_data(data, args.x_key, args.y_key)

    for split_name, (xs, ys) in data.items():
        plt.scatter(xs, ys, label=split_name)
        plot_line(xs, ys)

    plt.title(f"{args.y_key}")
    plt.xlabel(f"{args.x_key} ({args.x_scale})")
    plt.ylabel(args.y_key)

    plt.xscale(args.x_scale)

    add_legend(make_title(args.split_by), len(data))
    plt.grid()

    plt.show()


def make_title(key: str):
    title = " ".join(key.split(".")[-2:])
    title = title.replace("_", " ").capitalize()
    return title


def add_legend(title, entry_count):
    if 0 < entry_count <= 6:
        plt.legend(title=title + ":")
    if 6 < entry_count < 20:
        outside_legend(title=title + ":", fontsize="small")
        plt.tight_layout()


def print_data(data, x_key="x", y_key="y"):
    yd = {split: {x_key: xs, y_key: ys} for split, (xs, ys) in data.items()}
    print("\nData for plotting:")
    print(yaml.safe_dump(yd, default_flow_style=None))


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
