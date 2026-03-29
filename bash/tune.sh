YEAR = 2021

PYTHONPATH="$PWD:$PWD/src"  WANDB_MODE=online python src/experiments/optuna_tune.py \
  --model_config cfgs/rabina/utae_all_features.yaml \
  --data_config cfgs/data_multitemporal_full_features.yaml \
  --trainer_config cfgs/trainer_single_gpu.yaml \
  --data_dir "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_subset_hdf5" \
  --n_trials 10 \
  --study_name utae_optuna_1 \
  --storage sqlite:///optuna.db \
  --output_dir ./lightning_logs/optuna_runs \
  --lr_choices 1e-4 3e-4 1e-3 3e-3 1e-2 \
  --loss_choices BCE Dice Jaccard Focal Lovasz \
  --cross_year_split_json_path "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
  --cross_year_train_id $YEAR \
  --wandb_group optuna_experiment \
  --wandb_run_name utae_optuna_cross_year_${YEAR}_run \
  --do_cross_year_experiment true

#   --wandb_entity wildfire_continual_learning \
