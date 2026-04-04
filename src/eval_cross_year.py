import argparse
import csv
import json
import os
from pathlib import Path

from dataloader.FireSpreadDataModule import FireSpreadDataModule
from models import BaseModel
from train import MyLightningCLI
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint per train year against multiple test years and save a cross-year AP table."
    )
    parser.add_argument("--config", type=str, required=True, help="Base model config YAML.")
    parser.add_argument("--trainer", type=str, required=True, help="Trainer config YAML.")
    parser.add_argument("--data", type=str, required=True, help="Data config YAML.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root HDF5 dataset directory.")
    parser.add_argument("--split_json", type=str, required=True, help="JSON file with train/val/test event IDs per year.")
    parser.add_argument(
        "--checkpoint_csv",
        type=str,
        default=None,
        help="Optional CSV with columns: train_year,ckpt_path",
    )
    parser.add_argument(
        "--checkpoint_json",
        type=str,
        default=None,
        help="Optional JSON mapping train year to checkpoint path.",
    )
    parser.add_argument(
        "--train_years",
        type=str,
        required=True,
        help='Comma-separated train years, e.g. "2016,2017,2018"',
    )
    parser.add_argument(
        "--test_years",
        type=str,
        required=True,
        help='Comma-separated test years, e.g. "2016,2017,2018,2019,2020,2021,2022,2023"',
    )
    parser.add_argument("--output_csv", type=str, required=True, help="Where to save the result table CSV.")
    # parser.add_argument("--accelerator", type=str, default="cpu", help="Lightning accelerator override.")
    # parser.add_argument("--devices", type=str, default="1", help="Lightning devices override.")
    # parser.add_argument("--num_workers", type=str, default="0", help="DataLoader workers override.")
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable WandB during evaluation runs.",
    )
    return parser.parse_args()


# def load_checkpoint_map(path: Path):
#     mapping = {}
#     with open(path, "r", encoding="utf-8", newline="") as f:
#         reader = csv.DictReader(f)
#         required = {"train_year", "ckpt_path"}
#         missing = required.difference(reader.fieldnames or [])
#         if missing:
#             raise ValueError(
#                 f"checkpoint CSV missing required columns {sorted(missing)}; found {reader.fieldnames}"
#             )
#         for row in reader:
#             mapping[int(row["train_year"])] = row["ckpt_path"]
#     return mapping


def load_checkpoint_map_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("checkpoint JSON must be an object mapping train year to ckpt path.")

    mapping = {}
    for key, value in data.items():
        mapping[int(key)] = value
    return mapping


def build_cli_args(args, train_year: int, test_year: int, ckpt_path: str):
    return [
        "-c",
        args.config,
        "--trainer",
        args.trainer,
        "--data",
        args.data,
        "--data.data_dir",
        args.data_dir,
        "--do_train",
        "false",
        "--do_test",
        "true",
        "--do_validate",
        "false",
        "--do_predict",
        "false",
        "--ckpt_path",
        ckpt_path,
        "--data.do_cross_year_experiment",
        "true",
        "--data.cross_year_split_json_path",
        args.split_json,
        "--data.cross_year_train_id",
        str(train_year),
        "--data.cross_year_eval_id",
        str(test_year),
        # "--trainer.accelerator",
        # args.accelerator,
        # "--trainer.devices",
        # args.devices,
        # "--data.num_workers",
        # args.num_workers,
    ]


def evaluate_pair(cli_args):
    cli = MyLightningCLI(
        BaseModel,
        FireSpreadDataModule,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
        run=False,
        args=cli_args,
    )
    results = cli.trainer.test(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if not results:
        raise RuntimeError("trainer.test returned no results.")
    return results[0]


def main():
    args = parse_args()
    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    if bool(args.checkpoint_csv) == bool(args.checkpoint_json):
        raise ValueError("Provide exactly one of --checkpoint_csv or --checkpoint_json.")

    if args.checkpoint_json:
        checkpoint_map = load_checkpoint_map_json(
            Path(args.checkpoint_json).expanduser().resolve()
        )
    # else:
    #     checkpoint_map = load_checkpoint_map(
    #         Path(args.checkpoint_csv).expanduser().resolve()
    #     )
    train_years = [int(x.strip()) for x in args.train_years.split(",") if x.strip()]
    test_years = [int(x.strip()) for x in args.test_years.split(",") if x.strip()]

    metrics_tables = {
        "AP": [],
        "F1": [],
        "Precision": [],
        "Recall": [],
        "IoU": []
    }
    for train_year in train_years:
        if train_year not in checkpoint_map:
            raise KeyError(f"No checkpoint found for train_year={train_year} in {args.checkpoint_csv}")

        ckpt_path = checkpoint_map[train_year]

        # Initialize row for each metric
        row_ap = {"train_year": train_year}
        row_f1 = {"train_year": train_year}
        row_prec = {"train_year": train_year}
        row_rec = {"train_year": train_year}
        row_iou = {"train_year": train_year}

        for test_year in test_years:
            print(f"Evaluating train_year={train_year} on test_year={test_year}")

            cli_args = build_cli_args(args, train_year, test_year, ckpt_path)
            metrics = evaluate_pair(cli_args)

            row_ap[str(test_year)] = round(metrics.get("test_AP", 0), 6)
            row_f1[str(test_year)] = round(metrics.get("test_f1", 0), 6)
            row_prec[str(test_year)] = round(metrics.get("test_precision", 0), 6)
            row_rec[str(test_year)] = round(metrics.get("test_recall", 0), 6)
            row_iou[str(test_year)] = round(metrics.get("test_iou", 0), 6)

        # Compute averages
        for row, key in zip(
            [row_ap, row_f1, row_prec, row_rec, row_iou],
            ["AP", "F1", "Precision", "Recall", "IoU"]
        ):
            values = [row[str(year)] for year in test_years]
            row["Avg"] = sum(values) / len(values)
            metrics_tables[key].append(row)

    output_path = Path(args.output_csv).with_suffix(".xlsx").expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for metric_name, rows in metrics_tables.items():
            df_metric = pd.DataFrame(rows)
            df_metric.to_excel(writer, sheet_name=metric_name, index=False)

    print(f"Saved cross-year results to {output_path}")

if __name__ == "__main__":
    main()
