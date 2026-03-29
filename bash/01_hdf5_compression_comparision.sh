#!/bin/bash

# -----------------------
# RUN LOG
# -----------------------
# Year = 2019, Time taken to train 20 epoch
# Using raw tiff file: 24 min 10 sec
# Using hdf5 files: 9 min 6 sec
# Using hdf5 compressed file: 10 min 19 sec
# Optimum Choice would be using Compressed hdf5 file as training time is negligibly higher but stoarge is very optimum
# -----------------------

YEAR=2019
# -----------------------
# Original Datasets
# -----------------------
RUN_NAME="train_${YEAR}_raw_tiff"
echo "=====Running: ${RUN_NAME}========="

PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
  -c cfgs/rabina/utae_all_features.yaml \
  --trainer cfgs/trainer_single_gpu.yaml \
  --data cfgs/data_multitemporal_full_features.yaml \
  --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_all" \
  --trainer.max_epochs 20 \
  --model.init_args.encoder_weights none \
  --data.do_cross_year_experiment true \
  --data.load_from_hdf5 false \
  --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
  --data.cross_year_train_id "$YEAR" \
  --trainer.default_root_dir "./lightning_logs/hdf5_compression_comparison/${RUN_NAME}" \
  --trainer.logger.init_args.project "wildfire_progression" \
  --trainer.logger.init_args.entity "wildfire_continual_learning" \
  --trainer.logger.init_args.group "hdf5_compression_comparison" \
  --trainer.logger.init_args.name "${RUN_NAME}" \
  --trainer.logger.init_args.tags '["uncompressed"]'


# -----------------------
# UNCOMPRESSED HDF5
# -----------------------
RUN_NAME="train_${YEAR}_uncompressed"
echo "=====Running: ${RUN_NAME}========="

PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
  -c cfgs/rabina/utae_all_features.yaml \
  --trainer cfgs/trainer_single_gpu.yaml \
  --data cfgs/data_multitemporal_full_features.yaml \
  --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5" \
  --trainer.max_epochs 20 \
  --model.init_args.encoder_weights none \
  --data.do_cross_year_experiment true \
  --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
  --data.cross_year_train_id "$YEAR" \
  --trainer.default_root_dir "./lightning_logs/hdf5_compression_comparison/${RUN_NAME}" \
  --trainer.logger.init_args.project "wildfire_progression" \
  --trainer.logger.init_args.entity "wildfire_continual_learning" \
  --trainer.logger.init_args.group "hdf5_compression_comparison" \
  --trainer.logger.init_args.name "${RUN_NAME}" \
  --trainer.logger.init_args.id "${RUN_NAME}" \
  --trainer.logger.init_args.tags '["uncompressed"]'


# -----------------------
# 🔹 COMPRESSED HDF5
# -----------------------
RUN_NAME="train_${YEAR}_compressed"
echo "======== Running: ${RUN_NAME}========"
PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
  -c cfgs/rabina/utae_all_features.yaml \
  --trainer cfgs/trainer_single_gpu.yaml \
  --data cfgs/data_multitemporal_full_features.yaml \
  --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
  --trainer.max_epochs 20 \
  --model.init_args.encoder_weights none \
  --data.do_cross_year_experiment true \
  --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
  --data.cross_year_train_id "$YEAR" \
  --trainer.default_root_dir "./lightning_logs/hdf5_compression_comparison/${RUN_NAME}" \
  --trainer.logger.init_args.project "wildfire_progression" \
  --trainer.logger.init_args.entity "wildfire_continual_learning" \
  --trainer.logger.init_args.group "hdf5_compression_comparison" \
  --trainer.logger.init_args.name "${RUN_NAME}" \
  --trainer.logger.init_args.id "${RUN_NAME}" \
  --trainer.logger.init_args.tags '["compressed"]'