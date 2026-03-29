'''
PYTHONPATH="$PWD:$PWD/src" python src/experiments/optuna_tune.py \
  --model_config cfgs/UTAE/all_features.yaml \
  --data_config cfgs/data_multitemporal_full_features.yaml \
  --trainer_config cfgs/trainer_single_gpu.yaml \
  --data_dir "/absolute/path/to/WSTS_hdf5" \
  --n_trials 20 \
  --study_name utae_optuna \
  --storage sqlite:///optuna.db \
  --output_dir ./optuna_runs \
  --lr_choices 1e-4 3e-4 1e-3 3e-3 1e-2 \
  --loss_choices BCE Dice Jaccard
'''

import argparse
import os
from pathlib import Path
import optuna
from dataloader.FireSpreadDataModule import FireSpreadDataModule
from models import BaseModel
from train import MyLightningCLI
import pytorch_lightning as pl
import wandb

# pl.seed_everything(42)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter tuning with the existing LightningCLI training flow."
    )
    parser.add_argument("--model_config",required=True,help="Path to the base model config YAML, e.g. cfgs/UTAE/all_features.yaml",)
    parser.add_argument("--data_config",required=True,help="Path to the base data config YAML, e.g. cfgs/data_multitemporal_full_features.yaml",)
    parser.add_argument("--trainer_config",required=True, help="Path to the base trainer config YAML, e.g. cfgs/trainer_single_gpu.yaml",)
    parser.add_argument("--n_trials",type=int,default=20,help="Number of Optuna trials to run.",)
    parser.add_argument("--study_name",type=str,default="wildfire_optuna",help="Optuna study name.",)
    parser.add_argument("--storage",type=str,default=None,help="Optional Optuna storage URL, e.g. sqlite:///optuna.db",)
    parser.add_argument("--output_dir",type=str,required=True,default="./optuna_runs",help="Base directory for trial outputs.",)
    parser.add_argument("--data_dir",type=str,required=True,default=None,help="Optional override for data.data_dir.",)
    parser.add_argument("--lr_choices",nargs="+",type=float, default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],help="Learning rates to search over.",)
    parser.add_argument("--loss_choices",nargs="+",default=["BCE", "Dice", "Jaccard"],help="Loss functions to search over.",)
    # parser.add_argument("--do_cross_year_experiment",type=bool,default=False,help="Enable cross-year experiment",)
    parser.add_argument("--do_cross_year_experiment", type=str ,default = "false",help="Enable cross-year experiment")
    # parser.add_argument("--do_cross_year_experiment", action="store_true",help="Enable cross-year experiment")

    parser.add_argument("--cross_year_split_json_path",type=str,default=None,help="Path to cross-year split JSON file",)
    parser.add_argument("--cross_year_train_id",type=int,default=2016,help="Training year ID")

    # Trainer logger args
    parser.add_argument("--wandb_entity",type=str,default=None,help="WandB entity")
    parser.add_argument( "--wandb_group",type=str,default=None,help="WandB group name")
    parser.add_argument("--wandb_run_name",type=str,default=None,help="WandB run name")
    # parser.add_argument(
    #     "--encoder_weights",
    #     type=str,
    #     default=None,
    #     help="Optional override for model.init_args.encoder_weights, e.g. none or pastis.",
    # )
    # parser.add_argument(
    #     "--pretrained_checkpoint_path",
    #     type=str,
    #     default=None,
    #     help="Optional override for model.init_args.pretrained_checkpoint_path.",
    # )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_output_dir = Path(args.output_dir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    def build_cli_args(trial: "optuna.Trial") -> list[str]:
        trial_dir = base_output_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # lr = trial.suggest_float("optimizer.lr", args.lr_min, args.lr_max, log=True)
        lr = trial.suggest_categorical(
            "optimizer.lr", args.lr_choices
        )
        loss_function = trial.suggest_categorical(
            "model.init_args.loss_function", args.loss_choices
        )

        cli_args = [
            "-c",
            args.model_config,
            "--trainer",
            args.trainer_config,
            "--data",
            args.data_config,
            "--optimizer.init_args.lr",
            str(lr),
            "--model.init_args.loss_function",
            loss_function,
            # "--model.init_args.pos_class_weight",
            # "1.0",
            "--trainer.default_root_dir",
            str(trial_dir),
            "--data.data_dir", args.data_dir
        ]
        if args.do_cross_year_experiment is not None:
            cli_args.extend(["--data.do_cross_year_experiment", args.do_cross_year_experiment])
        if args.cross_year_split_json_path is not None:
            cli_args.extend(["--data.cross_year_split_json_path", args.cross_year_split_json_path])
        if args.cross_year_train_id is not None:
            cli_args.extend(["--data.cross_year_train_id", str(args.cross_year_train_id)])
        # if args.encoder_weights is not None:
        #     cli_args.extend(["--model.init_args.encoder_weights", str(args.encoder_weights)])
        # if args.pretrained_checkpoint_path is not None:
        #     cli_args.extend([
        #         "--model.init_args.pretrained_checkpoint_path",
        #         str(args.pretrained_checkpoint_path),
        #     ])

        if args.wandb_entity is not None:
            cli_args.extend(["--trainer.logger.init_args.entity", str(args.wandb_entity)])
        if args.wandb_group is not None:
            cli_args.extend(["--trainer.logger.init_args.group", str(args.wandb_group)])
        if args.wandb_run_name is not None:
            cli_args.extend(["--trainer.logger.init_args.name",f"{args.wandb_run_name}_trial_{trial.number}"])

        return cli_args

    def objective(trial: "optuna.Trial") -> float:
        
        if wandb.run is None: # RT: To prevent setup if wandb is disabled
            wandb.finish()
        
        # pl.seed_everything(42 + trial.number)
        cli_args = build_cli_args(trial)
        
        print(f"\n Starting trial {trial.number} with CLI args: {cli_args}")

        cli = MyLightningCLI(
            BaseModel,
            FireSpreadDataModule,
            subclass_mode_model=True,
            save_config_kwargs={"overwrite": True},
            parser_kwargs={"parser_mode": "yaml"},
            run=False,
            args=cli_args,
        )

        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt_path)

        print("train len:", len(cli.datamodule.train_dataset))
        print("val len:", len(cli.datamodule.val_dataset))

        metric = cli.trainer.callback_metrics.get("val_avg_precision")
        if metric is None:
            raise RuntimeError(
                "val_avg_precision was not found in trainer.callback_metrics after training."
            )

        score = float(metric.detach().cpu().item())
        trial.set_user_attr(
            "best_model_path",
            getattr(cli.trainer.checkpoint_callback, "best_model_path", ""),
        )
        return score

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest trial:")
    print(f"  value: {study.best_trial.value}")
    print("  params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    best_ckpt = study.best_trial.user_attrs.get("best_model_path", "")
    if best_ckpt:
        print(f" best_model_path: {best_ckpt}")


if __name__ == "__main__":
    main()
