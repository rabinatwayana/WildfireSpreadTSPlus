import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from dataloader.FireSpreadDataModule import FireSpreadDataModule
from models import BaseModel
from train import MyLightningCLI


class FullSegmentationWrapper(torch.nn.Module):
    """
    Wraps the segmentation model so SHAP explains one scalar per sample.
    Here we explain the mean predicted fire probability over the map.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x, doys=None):
        logits = self.model(x, doys)
        probs = torch.sigmoid(logits)

        # Reduce shapes like [B,1,H,W] or [B,1,1,H,W] to [B,H,W]
        while probs.ndim > 3 and probs.shape[1] == 1:
            probs = probs.squeeze(1)

        if probs.ndim != 3:
            raise ValueError(f"Expected [B, H, W] after squeezing, got {probs.shape}")

        out = probs.mean(dim=(1, 2))  # [B]
        return out.unsqueeze(1)       # [B, 1]


def get_positive_indices(dataset, size):
    pos_idx = []
    for i in range(len(dataset)):
        sample = dataset[i]
        y = sample[1]
        if (y > 0).any():
            pos_idx.append(i)
            if len(pos_idx) >= size:
                break
    return pos_idx


def take_batch(dataset, size, return_doy):
    xs, ys, ds = [], [], []
    pos_indices = get_positive_indices(dataset, size)
    print(f"Found {len(pos_indices)} positive samples in dataset of size {len(dataset)}")

    if len(pos_indices) == 0:
        raise RuntimeError("No positive samples found.")

    selected_idx = random.sample(pos_indices, min(size, len(pos_indices)))

    for i in selected_idx:
        sample = dataset[i]
        if return_doy:
            x, y, doy = sample
            ds.append(doy)
        else:
            x, y = sample
        xs.append(x)
        ys.append(y)

    x = torch.stack(xs)
    y = torch.stack(ys)

    if return_doy:
        d = torch.stack(ds)
        return x, y, d
    return x, y, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--do_cross_year_experiment", type=str, default="false")
    parser.add_argument("--cross_year_split_json_path", type=str, default=None)
    parser.add_argument("--cross_year_train_id", type=int, default=2016)
    parser.add_argument("--background_size", type=int, default=1)
    parser.add_argument("--explain_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./shap_feature_plots")
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    start_time = time.perf_counter()
    args, unknown = parse_args()

    cli_args = [
        "-c", args.config,
        "--trainer", args.trainer,
        "--data", args.data,
        "--data.data_dir", args.data_dir,
        "--data.do_cross_year_experiment", args.do_cross_year_experiment,
        "--data.cross_year_split_json_path", args.cross_year_split_json_path,
        "--data.cross_year_train_id", str(args.cross_year_train_id),
        "--do_train", "false",
        "--do_validate", "false",
        "--do_test", "false",
        "--do_predict", "false",
        "--ckpt_path", args.ckpt_path,
    ] + unknown

    cli = MyLightningCLI(
        BaseModel,
        FireSpreadDataModule,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
        run=False,
        args=cli_args,
    )

    print("CLI initialized.")
    print(cli.config)
    print("Args used:", args)

    cli.datamodule.setup(stage="fit")
    train_dataset = cli.datamodule.train_dataset
    print("Train dataset size:", len(train_dataset))

    cli.datamodule.setup(stage="test")
    test_dataset = cli.datamodule.test_dataset
    print("Test dataset size:", len(test_dataset))

    # Force CPU for SHAP
    device = torch.device("cpu")

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    cli.model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = cli.model.to(device).eval()

    print("Device used:", next(model.parameters()).device)

    return_doy = bool(cli.config.data.return_doy)

    bg_x, _, bg_d = take_batch(train_dataset, args.background_size, return_doy)
    ex_x, _, ex_d = take_batch(test_dataset, args.explain_size, return_doy)

    print("Background batch shape:", bg_x.shape, "DOY shape:", bg_d.shape if bg_d is not None else "N/A")
    print("Explanation batch shape:", ex_x.shape, "DOY shape:", ex_d.shape if ex_d is not None else "N/A")

    bg_x = bg_x.to(device)
    ex_x = ex_x.to(device)

    if return_doy:
        bg_d = bg_d.to(device)
        ex_d = ex_d.to(device)

    wrapper = FullSegmentationWrapper(model).to(device)

    # GradientExplainer on CPU
    if return_doy:
        explainer = shap.GradientExplainer(
            lambda x, d: wrapper(x, d),
            [bg_x, bg_d],
        )
        shap_values = explainer.shap_values([ex_x, ex_d])
    else:
        explainer = shap.GradientExplainer(wrapper, bg_x)
        shap_values = explainer.shap_values(ex_x)

    print("Raw SHAP type:", type(shap_values))

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values)
    print("Raw SHAP shape:", shap_values.shape)

    # Expected [B, T, F, H, W, 1] or [B, T, F, H, W]
    if shap_values.ndim == 6 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    if shap_values.ndim != 5:
        raise ValueError(f"Expected SHAP shape [B, T, F, H, W], got {shap_values.shape}")

    print("Final SHAP shape:", shap_values.shape)

    shap_vals = shap_values[0]  # [T, F, H, W]
    T, F, H, W = shap_vals.shape

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Overall feature importance
    feature_importance = np.sum(np.abs(shap_vals), axis=(0, 2, 3))  # [F]
    denom = feature_importance.sum()
    if denom > 0:
        feature_importance = feature_importance / denom

    plt.figure(figsize=(12, 6))
    plt.bar(range(F), feature_importance)
    plt.xlabel("Feature index")
    plt.ylabel("Normalized SHAP importance")
    plt.title("Overall feature importance (sum over time & pixels)")
    plt.savefig(os.path.join(args.save_dir, "feature_importance.png"))
    plt.close()

    # 2. Feature importance per time step
    feature_time = np.sum(np.abs(shap_vals), axis=(2, 3))  # [T, F]

    plt.figure(figsize=(12, 6))
    for f in range(F):
        plt.plot(range(T), feature_time[:, f], label=f"Feature {f}")
    plt.xlabel("Time step")
    plt.ylabel("SHAP importance")
    plt.title("Feature importance per time step")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "feature_importance_per_time.png"))
    plt.close()

    # 3. Top feature per pixel per time step
    top_feature_idx = np.argmax(np.abs(shap_vals), axis=1)  # [T, H, W]

    for t in range(T):
        plt.figure(figsize=(6, 6))
        plt.imshow(top_feature_idx[t], cmap="tab20")
        plt.colorbar()
        plt.title(f"Top feature per pixel - time step {t}")
        plt.savefig(os.path.join(args.save_dir, f"top_feature_per_pixel_t{t}.png"))
        plt.close()

    print(f"SHAP feature plots saved to {args.save_dir}/")
    print(f"Total time taken: {time.perf_counter() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
