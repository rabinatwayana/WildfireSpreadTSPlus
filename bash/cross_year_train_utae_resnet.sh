#!/bin/bash -l
export PYTHONPATH="$PWD:$PWD/src"
export WANDB_MODE=online

# ==================================================================
# l2-l3spatial based for cross year experiment
# ==================================================================
ENTITY="wildfire_continual_learning"
GROUP_NAME=cross_year_train_utae_resnet_ecol2l3_debug
LIGHTENING_LOG_DIR="./lightning_logs/cross_year_train_utae_resnet_ecol2l3_debug"

# for YEAR in 2016 2017 2018 2019 2020 2021 2022 2023; do
for YEAR in 2019; do
# RUN_NAME="utae_resnet_train_${YEAR}_focaldice0.0001_crop192_128stats" try for 192 stats
  RUN_NAME="utae_resnet_tr_${YEAR}_fo0.75di0.0001_cr128_43feat_imgnet_ofload"
  echo "======================================"
  echo "STARTING YEAR: $YEAR"
  echo "RUN NAME: $RUN_NAME"
  echo "======================================"
  python src/train.py \
    -c cfgs/rabina/utae_resnet_all_features.yaml \
    --trainer cfgs/trainer_single_gpu.yaml \
    --data cfgs/data_multitemporal_full_features.yaml \
    --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized_cropped_128" \
    --trainer.max_epochs 100 \
    --data.do_cross_year_experiment true \
    --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_spatial_eco_l2l3_split.json" \
    --data.cross_year_train_id "$YEAR" \
    --trainer.default_root_dir "${LIGHTENING_LOG_DIR}/${RUN_NAME}" \
    --trainer.logger.init_args.entity "${ENTITY}" \
    --trainer.logger.init_args.group "${GROUP_NAME}" \
    --trainer.logger.init_args.name "${RUN_NAME}" \
    --data.batch_size 16
done

    # --model.init_args.loss_function "BCE+Dice" \

# --model.init_args.loss_function "BCE+Dice" \
#     --optimizer.init_args.lr 1e-3 \

    # --trainer.precision 16
#  --data.batch_size 32
    # --model.init_args.encoder_weights none \
# 2018 2019 2020 2021
# --data.batch_size 4
    # --model.init_args.n_channels 43 \
    # --trainer.precision 32 \
# 
