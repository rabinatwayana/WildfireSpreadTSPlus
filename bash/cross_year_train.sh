# duration based stratified randon split for cross year experiment
# ENTITY="wildfire_continual_learning"
# GROUP_NAME=cross_year_train
# LIGHTENING_LOG_DIR="./lightning_logs/cross_year_train"

# for YEAR in 2016 2017 2018 2019 2020 2021 2022 2023; do
#   RUN_NAME="utae_train_${YEAR}"
#   PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
#     -c cfgs/rabina/utae_all_features.yaml \
#     --trainer cfgs/trainer_single_gpu.yaml \
#     --data cfgs/data_multitemporal_full_features.yaml \
#     --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
#     --trainer.max_epochs 100 \
#     --data.do_cross_year_experiment true \
#     --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
#     --data.cross_year_train_id "$YEAR" \
#     --trainer.default_root_dir "${LIGHTENING_LOG_DIR}/${RUN_NAME}" \
#     --trainer.logger.init_args.entity "${ENTITY}" \
#     --trainer.logger.init_args.group "${GROUP_NAME}" \
#     --trainer.logger.init_args.name "${RUN_NAME}"
# done


# spatial based for cross year experiment
ENTITY="wildfire_continual_learning"
GROUP_NAME=cross_year_train_spatial_split
LIGHTENING_LOG_DIR="./lightning_logs/cross_year_train_spatial_split"

for YEAR in 2018; do
  RUN_NAME="utae_train_${YEAR}"
  PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
    -c cfgs/rabina/utae_all_features.yaml \
    --trainer cfgs/trainer_single_gpu.yaml \
    --data cfgs/data_multitemporal_full_features.yaml \
    --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
    --trainer.max_epochs 100 \
    --data.do_cross_year_experiment true \
    --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_spatial_eco_l3_split.json" \
    --data.cross_year_train_id "$YEAR" \
    --trainer.default_root_dir "${LIGHTENING_LOG_DIR}/${RUN_NAME}" \
    --trainer.logger.init_args.entity "${ENTITY}" \
    --trainer.logger.init_args.group "${GROUP_NAME}" \
    --trainer.logger.init_args.name "${RUN_NAME}"
done

    # --model.init_args.encoder_weights none \
# for YEAR in 2016,2017,2018,2019,2020,2021,2022,2023; do

