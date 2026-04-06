#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc

# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
# SBATCH --gres gpu:1
# SBATCH --constraint 2080
# SBATCH --cpus-per-gpu 10
# SBATCH --mem 64G
# SBATCH --nodelist=sn1

# set -euo pipefail
# source /share/common/anaconda3/etc/profile.d/conda.sh
# conda activate cde_mt_3.12

export PYTHONPATH="$PWD:$PWD/src"
export WANDB_MODE=online
# export WANDB_DEBUG=true
# unset WANDB_DISABLED
# # Set this if your cluster does not automatically pass it into sbatch jobs
# export WANDB_API_KEY="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"

# ==================================================================
# duration based stratified randon split for cross year experiment
# ==================================================================
# ENTITY="wildfire_continual_learning"
# GROUP_NAME=cross_year_train
# LIGHTENING_LOG_DIR="./lightning_logs/cross_year_train"

# for YEAR in 2018; do
#   RUN_NAME="utae_train_${YEAR}_doy"
#   PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
#     -c cfgs/rabina/utae_all_features.yaml \
#     --trainer cfgs/trainer_single_gpu.yaml \
#     --data cfgs/data_multitemporal_full_features.yaml \
#     --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
#     --trainer.max_epochs 100 \
#     --model.init_args.temporal_position_mode "doy" \
#     --data.return_doy true \
#     --data.do_cross_year_experiment true \
#     --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
#     --data.cross_year_train_id "$YEAR" \
#     --trainer.default_root_dir "${LIGHTENING_LOG_DIR}/${RUN_NAME}" \
#     --trainer.logger.init_args.entity "${ENTITY}" \
#     --trainer.logger.init_args.group "${GROUP_NAME}" \
#     --trainer.logger.init_args.name "${RUN_NAME}"
# done

# --model.init_args.loss_function "Dice" \

# ==================================================================
# l3-eco spatial based for cross year experiment
# ==================================================================

# ENTITY="wildfire_continual_learning"
# GROUP_NAME=cross_year_train_spatial_split
# LIGHTENING_LOG_DIR="./lightning_logs/cross_year_train_spatial_split"

# for YEAR in 2018; do
#   RUN_NAME="utae_train_${YEAR}"
#   PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
#     -c cfgs/rabina/utae_all_features.yaml \
#     --trainer cfgs/trainer_single_gpu.yaml \
#     --data cfgs/data_multitemporal_full_features.yaml \
#     --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
#     --trainer.max_epochs 100 \
#     --data.do_cross_year_experiment true \
#     --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_spatial_eco_l3_split.json" \
#     --data.cross_year_train_id "$YEAR" \
#     --trainer.default_root_dir "${LIGHTENING_LOG_DIR}/${RUN_NAME}" \
#     --trainer.logger.init_args.entity "${ENTITY}" \
#     --trainer.logger.init_args.group "${GROUP_NAME}" \
#     --trainer.logger.init_args.name "${RUN_NAME}"
# done

# ==================================================================
# l2-l3spatial based for cross year experiment
# ==================================================================
ENTITY="wildfire_continual_learning"
GROUP_NAME=cross_year_train_ecol2l3_debug
LIGHTENING_LOG_DIR="./lightning_logs/cross_year_train_ecol2l3_debug"

# for YEAR in 2016 2017 2018 2019 2020 2021 2022 2023; do
for YEAR in 2018; do
  RUN_NAME="utae_train_${YEAR}_posweight20_bce_dice"
  echo "======================================"
  echo "STARTING YEAR: $YEAR"
  echo "RUN NAME: $RUN_NAME"
  echo "======================================"
  python src/train.py \
    -c cfgs/rabina/utae_all_features.yaml \
    --trainer cfgs/trainer_single_gpu.yaml \
    --data cfgs/data_multitemporal_full_features.yaml \
    --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
    --trainer.max_epochs 100 \
    --data.do_cross_year_experiment true \
    --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_spatial_eco_l2l3_split.json" \
    --data.cross_year_train_id "$YEAR" \
    --trainer.default_root_dir "${LIGHTENING_LOG_DIR}/${RUN_NAME}" \
    --trainer.logger.init_args.entity "${ENTITY}" \
    --trainer.logger.init_args.group "${GROUP_NAME}" \
    --trainer.logger.init_args.name "${RUN_NAME}" \
    --model.init_args.loss_function "BCE" \
    --optimizer.init_args.lr 1e-3 \
    --data.batch_size 16
done
    # --trainer.precision 16
#  --data.batch_size 32
    # --model.init_args.encoder_weights none \
# 2018 2019 2020 2021
# --data.batch_size 4
    # --model.init_args.n_channels 43 \
    # --trainer.precision 32 \
# 
