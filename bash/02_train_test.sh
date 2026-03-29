ENTITY="wildfire_continual_learning"
GROUP_NAME=TEST
LIGHTENING_LOG_DIR="./lightning_logs/02_test"

# Using weight=None
# for YEAR in 2016; do
#   RUN_NAME="utae_train_${YEAR}"
#   PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
#     -c cfgs/rabina/utae_all_features.yaml \
#     --trainer cfgs/trainer_single_gpu.yaml \
#     --data cfgs/data_multitemporal_full_features.yaml \
#     --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
#     --trainer.max_epochs 20 \
#     --model.init_args.encoder_weights none \
#     --data.do_cross_year_experiment true \
#     --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
#     --data.cross_year_train_id "$YEAR" \
#     --trainer.default_root_dir "${LIGHTENING_LOG_DIR}/${RUN_NAME}" \
#     --trainer.logger.init_args.entity "${ENTITY}" \
#     --trainer.logger.init_args.group "${GROUP_NAME}" \
#     --trainer.logger.init_args.name "${RUN_NAME}"
# done


# Using pastis weight
for YEAR in 2016; do
  RUN_NAME="utae_train_pastis_${YEAR}"
  PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
    -c cfgs/rabina/utae_all_features.yaml \
    --trainer cfgs/trainer_single_gpu.yaml \
    --data cfgs/data_multitemporal_full_features.yaml \
    --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
    --trainer.max_epochs 20 \
    --data.do_cross_year_experiment true \
    --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
    --data.cross_year_train_id "$YEAR" \
    --trainer.default_root_dir "${LIGHTENING_LOG_DIR}/${RUN_NAME}" \
    --trainer.logger.init_args.entity "${ENTITY}" \
    --trainer.logger.init_args.group "${GROUP_NAME}" \
    --trainer.logger.init_args.name "${RUN_NAME}"
done