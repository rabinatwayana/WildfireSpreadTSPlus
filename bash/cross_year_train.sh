for YEAR in 2018; do
  RUN_NAME="utae_trainv_${YEAR}"
  PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
    -c cfgs/UTAE/all_features_dumb_doy.yaml \
    --trainer cfgs/trainer_single_gpu.yaml \
    --data cfgs/data_multitemporal_full_features.yaml \
    --data.data_dir "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_subset_hdf5" \
    --trainer.accelerator cpu \
    --trainer.devices 1 \
    --data.num_workers 0 \
    --trainer.max_epochs 2 \
    --trainer.limit_train_batches 4 \
    --trainer.limit_val_batches 2 \
    --trainer.limit_test_batches 2 \
    --model.init_args.loss_function BCE \
    --model.init_args.encoder_weights none \
    --data.do_cross_year_experiment true \
    --data.cross_year_split_json_path "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split_test.json" \
    --data.cross_year_train_id "$YEAR" \
    --trainer.default_root_dir "./lightning_logs/cross_year_train/${RUN_NAME}" \
    --trainer.logger.init_args.name "${RUN_NAME}" \
    --trainer.logger.init_args.group "cross_year_train"
    # --trainer.logger.init_args.id "${RUN_NAME}" \
done