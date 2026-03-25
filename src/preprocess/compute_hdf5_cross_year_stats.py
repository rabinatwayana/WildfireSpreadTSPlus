'''
python src/preprocess/compute_hdf5_cross_year_stats.py \
  --data_dir "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_subset_hdf5" \
  --years "2016,2017,2018,2019,2020,2021,2022,2023" \
  --split_csv "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/output/data_split/wsts_combined_duration_stratified_split.csv" \
  --split_column "duration_based_split" \
  --split_value "train" \
  --output "/Users/rabinatwayana/2_CDE_MT/WildfireSpreadTSPlus/stats/train_stats_by_year.json"
  
'''
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


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
        "--split_csv",
        type=str,
        required=True,
        help="CSV containing at least year, event_id, and split columns.",
    )
    parser.add_argument(
        "--split_column",
        type=str,
        default="duration_based_split",
        help="Column name used to filter rows in --split_csv.",
    )
    parser.add_argument(
        "--split_value",
        type=str,
        default="train",
        help="Value in --split_column to keep when using --split_csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to save results as JSON.",
    )
    return parser.parse_args()

def get_selected_hdf5_files(data_dir, df):
    file_paths = []
    for _, row in df.iterrows():
        year = row["year"]
        event_id = row["event_id"]
        h5_path = data_dir / str(year) / f"{event_id}.hdf5"
        if not h5_path.exists():
            print(f"Warning: HDF5 file not found, skipping: {h5_path}")
            continue
        file_paths.append(h5_path)
    return file_paths


def compute_stats_for_files(h5_paths):
    print(h5_paths,"=============")
    sum_vals = None
    sum_sq_vals = None
    valid_counts = None
    missing_counts = None
    total_counts = None
    num_files = 0

    for h5_path in h5_paths:
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

        reshaped = np.moveaxis(data, 1, 0).reshape(n_channels, -1) #reorders the axes, and flattened per channel into rows.
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
        out=np.full_like(sum_vals, np.nan, dtype=np.float64), #output array
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
        "num_files": num_files,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "missing_values": missing_rates.tolist(),
        # "valid_counts": valid_counts.tolist(),
        # "missing_counts": missing_counts.tolist(),
        # "total_counts": total_counts.tolist(),
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir) 

    split_csv_path = Path(args.split_csv)

    df = pd.read_csv(split_csv_path)
    required = {"year", "event_id", args.split_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
    
    results=[]
    years = [int(y) for y in args.years.split(",")]
    for year in years:
        selected_df = df[
                (df["year"] == int(year)) &
                (df[args.split_column] == args.split_value)
            ]
        selected_files= get_selected_hdf5_files(data_dir, selected_df)

        stats = compute_stats_for_files(selected_files)
        stats["years"] = [year]
        stats["split_column"] = args.split_column
        stats["split_value"] = args.split_value
        results.append(stats)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved stats to {output_path}")

if __name__ == "__main__":
    main()
