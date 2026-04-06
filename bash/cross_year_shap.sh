#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=disabled python src/explain_deep_shap.py \
  -c cfgs/rabina/utae_all_features.yaml \
  --trainer cfgs/trainer_single_gpu.yaml \
  --data cfgs/data_multitemporal_full_features.yaml \
  --ckpt_path "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train_ecol2l3_debug/utae_train_2018_posweight20/wildfire_progression/jqak9tz2/checkpoints/best-epoch=42-val_avg_precision=0.43.ckpt" \
  --data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
  --do_cross_year_experiment true \
  --cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_spatial_eco_l3_split.json" \
  --cross_year_train_id 2018 \
  --background_size 200 \
  --explain_size 100 \
  --save_dir "results/shap_feature_plots/cross_year_ecol2l3_debug/2018_posweight18"

  # bash bash/cross_year_shap.sh > shap_feature_plots/2018_posweight18_test.log 2>&1
  # --ckpt_path "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train_ecol2l3_split/utae_train_2016/wildfire_progression/5gjc7zy3/checkpoints/best-epoch=52-val_avg_precision=0.20.ckpt" \

  