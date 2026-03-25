
'''
python3 src/preprocess/compute_hdf5_stats.py \
  --data_dir "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_subset_hdf5" \
  --years "2018,2019" \
  --output "/tmp/stats_2018_2019.json"
'''


import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean, std, and missing-value rate from WSTS/WSTS+ HDF5 files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing year-wise HDF5 folders.",
    )
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help='Comma-separated years to include, e.g. "2018,2019" or "2016,2017,2020,2021".',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )
    return parser.parse_args()


def iter_hdf5_files(data_dir: Path, years: list[int]):
    for year in years:
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            print(f"Warning: year directory not found, skipping: {year_dir}")
            continue
        for h5_path in sorted(year_dir.glob("*.hdf5")):
            yield h5_path


def compute_stats(data_dir: Path, years: list[int]):
    sum_vals = None
    sum_sq_vals = None
    valid_counts = None
    missing_counts = None
    total_counts = None
    num_files = 0

    for h5_path in iter_hdf5_files(data_dir, years):
        with h5py.File(h5_path, "r") as f:
            data = f["data"][:]  # shape: [time, channels, H, W]

        if data.ndim != 4:
            raise ValueError(f"Expected 4D data in {h5_path}, got shape {data.shape}")

        num_files += 1
        n_channels = data.shape[1]

        if sum_vals is None:
            sum_vals = np.zeros(n_channels, dtype=np.float64)
            sum_sq_vals = np.zeros(n_channels, dtype=np.float64)
            valid_counts = np.zeros(n_channels, dtype=np.int64)
            missing_counts = np.zeros(n_channels, dtype=np.int64)
            total_counts = np.zeros(n_channels, dtype=np.int64)

        reshaped = np.moveaxis(data, 1, 0).reshape(n_channels, -1)
        missing_mask = np.isnan(reshaped)
        valid_mask = ~missing_mask

        missing_counts += missing_mask.sum(axis=1)
        total_counts += reshaped.shape[1]

        valid_values = np.where(valid_mask, reshaped, 0.0)
        sum_vals += valid_values.sum(axis=1)
        sum_sq_vals += np.square(valid_values).sum(axis=1)
        valid_counts += valid_mask.sum(axis=1)

    if num_files == 0:
        raise FileNotFoundError("No HDF5 files found for the requested years.")

    means = np.divide(
        sum_vals,
        valid_counts,
        out=np.full_like(sum_vals, np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    second_moment = np.divide(
        sum_sq_vals,
        valid_counts,
        out=np.full_like(sum_sq_vals, np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    variances = second_moment - np.square(means)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances)
    missing_rates = np.divide(
        missing_counts,
        total_counts,
        out=np.full_like(sum_vals, np.nan, dtype=np.float64),
        where=total_counts > 0,
    )

    return {
        "years": years,
        "num_files": num_files,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "missing_values": missing_rates.tolist(),
        "valid_counts": valid_counts.tolist(),
        "missing_counts": missing_counts.tolist(),
        "total_counts": total_counts.tolist(),
    }


def main():
    args = parse_args()
    years = [int(year.strip()) for year in args.years.split(",") if year.strip()]
    data_dir = Path(args.data_dir).expanduser().resolve()

    stats = compute_stats(data_dir, years)
    print(json.dumps(stats, indent=2))

    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats to {output_path}")


if __name__ == "__main__":
    main()
