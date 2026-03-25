"""
python src/preprocess/format_hdf5_stats_for_utils.py \
  --input "/Users/rabinatwayana/2_CDE_MT/WildfireSpreadTSPlus/tmp/train_stats_by_year.json"
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Format computed HDF5 stats into the utils.py dictionary style."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the JSON output from compute_hdf5_stats.py",
    )
    return parser.parse_args()


def format_array(name: str, values: list[float]) -> str:
    lines = [f'"{name}": np.array([']
    for value in values:
        lines.append(f"    {repr(float(value))},")
    lines.append("], dtype=np.float32)")
    return "\n".join(lines)


def format_key(years: list[int]) -> str:
    if len(years) == 1:
        return f"({years[0]},)"
    return "(" + ", ".join(str(year) for year in years) + ")"


def print_entry(stats: dict):
    years = stats["years"]
    print(f"{format_key(years)}: {{")
    print(format_array("means", stats["means"]) + ",")
    print(format_array("stds", stats["stds"]) + ",")
    print(format_array("missing_values", stats["missing_values"]))
    print("},")


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()

    with open(input_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if isinstance(stats, list):
        for idx, entry in enumerate(stats):
            if idx > 0:
                print()
            print_entry(entry)
    else:
        print_entry(stats)


if __name__ == "__main__":
    main()
