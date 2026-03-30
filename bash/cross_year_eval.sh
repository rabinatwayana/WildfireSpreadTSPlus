# bash bash/cross_year_eval.sh > results/utae_cross_year_eval.log 2>&1

PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=disabled python src/eval_cross_year.py \
  --config cfgs/rabina/utae_all_features.yaml \
  --trainer cfgs/trainer_single_gpu.yaml \
  --data cfgs/data_multitemporal_full_features.yaml \
  --data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
  --split_json "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
  --checkpoint_json "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/inputs/utae_cross_year_checkpoints.json" \
  --train_years "2016,2017,2018,2019,2020,2021,2022,2023" \
  --test_years "2016,2017,2018,2019,2020,2021,2022,2023" \
  --output_csv "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/results/utae_cross_year_evaluation.csv" \
  --disable_wandb



## not used and not tested yet
# YEARS=(2018 2019)
# ENTITY="wildfire_continual_learning"
# GROUP_NAME=cross_year_eval_del

# declare -A CKPTS
# CKPTS[2016]="/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_train_2016/wildfire_progression/sb81gofz/checkpoints/best-epoch=9-val_avg_precision=0.01.ckpt"
# CKPTS[2017]="/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_train_2017/wildfire_progression/5jy319f4/checkpoints/best-epoch=40-val_avg_precision=0.43.ckpt"
# CKPTS[2018]="/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_train_2018/wildfire_progression/vwl3bnk4/checkpoints/best-epoch=19-val_avg_precision=0.59.ckpt"
# CKPTS[2019]="/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_train_2019/wildfire_progression/p0n13f8n/checkpoints/best-epoch=17-val_avg_precision=0.35.ckpt"
# CKPTS[2020]="/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_train_2020/wildfire_progression/7a9wnl0b/checkpoints/best-epoch=18-val_avg_precision=0.39.ckpt"
# CKPTS[2021]="/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_train_2021/wildfire_progression/om99y8t6/checkpoints/best-epoch=69-val_avg_precision=0.65.ckpt"
# CKPTS[2022]="/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_train_2022/wildfire_progression/0zngb9tk/checkpoints/best-epoch=10-val_avg_precision=0.00.ckpt"
# CKPTS[2023]="/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/lightning_logs/cross_year_train/utae_train_2023/wildfire_progression/8bycz1b7/checkpoints/best-epoch=10-val_avg_precision=0.00.ckpt"

# for TRAIN_YEAR in "${YEARS[@]}"
# do
#   for TEST_YEAR in "${YEARS[@]}"
#   do
#     echo "Train: $TRAIN_YEAR → Test: $TEST_YEAR"

#     PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/experiments/eval_cross_year_table.py \
#       -c cfgs/rabina/utae_all_features.yaml \
#       --trainer cfgs/trainer_single_gpu.yaml \
#       --data cfgs/data_multitemporal_full_features.yaml \
#       --data.data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
#       --data.do_cross_year_experiment true \
#       --data.cross_year_split_json_path "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
#       --data.cross_year_train_id $TRAIN_YEAR \
#       --data.cross_year_eval_id $TEST_YEAR \
#       --do_train false \
#       --do_validate false \
#       --do_test true \
#       --ckpt_path "${CKPTS[$TRAIN_YEAR]}" \
#       --trainer.default_root_dir "./lightning_logs/cross_year_eval_del/train_${TRAIN_YEAR}_test_${TEST_YEAR}" \
#       --trainer.logger.init_args.entity "${ENTITY}" \
#       --trainer.logger.init_args.group "${GROUP_NAME}" \
#       --trainer.logger.init_args.name "utae_train_${TRAIN_YEAR}_test_${TEST_YEAR}"
#   done
# done

#       # --trainer.accelerator cpu \
#       # --trainer.devices 1 \
#       # --data.num_workers 0 \
#       # --trainer.limit_test_batches 2 \
#       # --model.init_args.loss_function BCE \
#       # --model.init_args.encoder_weights none \
