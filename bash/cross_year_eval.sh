YEARS=(2016 2017)
GROUP_NAME="utae_allfeat_cross_year_eval"

declare -A CKPTS
CKPTS[2016]="/Users/rabinatwayana/2_CDE_MT/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_trainv_2018/wildfire_progression/utae_trainv_2018/checkpoints/best-epoch=1-val_avg_precision=0.01.ckpt"
CKPTS[2017]="/Users/rabinatwayana/2_CDE_MT/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_trainv_2018/wildfire_progression/utae_trainv_2018/checkpoints/best-epoch=1-val_avg_precision=0.01.ckpt"
# CKPTS[2018]="path/to/ckpt_2018.ckpt"
# CKPTS[2019]="path/to/ckpt_2019.ckpt"
# CKPTS[2020]="path/to/ckpt_2020.ckpt"
# CKPTS[2021]="path/to/ckpt_2021.ckpt"
# CKPTS[2022]="path/to/ckpt_2022.ckpt"
# CKPTS[2023]="path/to/ckpt_2023.ckpt"

for TRAIN_YEAR in "${YEARS[@]}"
do
  for TEST_YEAR in "${YEARS[@]}"
  do
    echo "Train: $TRAIN_YEAR → Test: $TEST_YEAR"

    PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
      -c cfgs/UTAE/all_features_dumb_doy.yaml \
      --trainer cfgs/trainer_single_gpu.yaml \
      --data cfgs/data_multitemporal_full_features.yaml \
      --data.data_dir "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_subset_hdf5" \
      --trainer.accelerator cpu \
      --trainer.devices 1 \
      --data.num_workers 0 \
      --trainer.limit_test_batches 2 \
      --model.init_args.loss_function BCE \
      --model.init_args.encoder_weights none \
      --data.do_cross_year_experiment true \
      --data.cross_year_split_json_path "/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split_test.json" \
      --data.cross_year_train_id $TRAIN_YEAR \
      --data.cross_year_eval_id $TEST_YEAR \
      --do_train false \
      --do_validate false \
      --do_test true \
      --ckpt_path "${CKPTS[$TRAIN_YEAR]}" \
      --trainer.default_root_dir "./lightning_logs/cross_year_eval/train_${TRAIN_YEAR}_test_${TEST_YEAR}" \
      --trainer.logger.init_args.name "train_${TRAIN_YEAR}_test_${TEST_YEAR}" \
      --trainer.logger.init_args.group "${GROUP_NAME}"
    #   --trainer.logger.init_args.id "${RUN_NAME}" \

  done
done