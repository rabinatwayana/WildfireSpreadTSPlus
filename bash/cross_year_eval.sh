# bash bash/cross_year_eval.sh > results/utae_cross_year_eval_wstsplus_dice.log 2>&1

# cross-year eval using l2l3 based spatial split , Note: lr and loss is updated to match the training config used for this experiment
PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=disabled python src/eval_cross_year.py \
  --config cfgs/rabina/utae_all_features.yaml \
  --trainer cfgs/trainer_single_gpu.yaml \
  --data cfgs/data_multitemporal_full_features.yaml \
  --data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
  --split_json "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_spatial_eco_l2l3_split.json" \
  --checkpoint_json "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/inputs/3_utae_cross_year_spatial_ecol2l3.json" \
  --train_years "2016,2017,2018,2019,2020,2021,2022,2023" \
  --test_years "2016,2017,2018,2019,2020,2021,2022,2023" \
  --output_csv "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/results/3_utae_cross_year_evaluation_ecol2l3.csv" \
  --disable_wandb

# # cross-year eval in only wsts+ years with dice loss
# PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=disabled python src/eval_cross_year.py \
#   --config cfgs/rabina/utae_all_features.yaml \
#   --trainer cfgs/trainer_single_gpu.yaml \
#   --data cfgs/data_multitemporal_full_features.yaml \
#   --data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
#   --split_json "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
#   --checkpoint_json "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/inputs/2_utae_cross_year_checkpoints_dice.json" \
#   --train_years "2016,2017,2022,2023" \
#   --test_years "2016,2017,2018,2019,2020,2021,2022,2023" \
#   --output_csv "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/results/2_utae_cross_year_evaluation_wstsplus_dice.csv" \
#   --disable_wandb

# cross-year eval in all years with focal loss  
# PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=disabled python src/eval_cross_year.py \
#   --config cfgs/rabina/utae_all_features.yaml \
#   --trainer cfgs/trainer_single_gpu.yaml \
#   --data cfgs/data_multitemporal_full_features.yaml \
#   --data_dir "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/data/processed/WSTS_all_hdf5_optimized" \
#   --split_json "/share/home/e2406754/cde_mt/CDE_Master_Thesis_Wildfire_CL/output/data_split/cross_year_duration_stratified_split.json" \
#   --checkpoint_json "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/inputs/1_utae_cross_year_checkpoints.json" \
#   --train_years "2016,2017,2018,2019,2020,2021,2022,2023" \
#   --test_years "2016,2017,2018,2019,2020,2021,2022,2023" \
#   --output_csv "/share/home/e2406754/cde_mt/WildfireSpreadTSPlus/results/1_utae_cross_year_evaluation.csv" \
#   --disable_wandb


