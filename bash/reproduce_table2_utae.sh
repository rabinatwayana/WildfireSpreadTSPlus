#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_subset_hdf5"
OUTPUT_DIR="./runs/table2_utae"
GROUP_NAME="table2_utae"

# export PYTHONPATH="${PYTHONPATH:-}:$PWD:$PWD/src"

mkdir -p "$OUTPUT_DIR"

for feature_set in veg multi all; do
  for fold in $(seq 0 11); do

    # ✅ Replace associative array with case
    case $feature_set in
      veg)
        FEATURES='[0, 1, 2, 3, 4, 38, 39]'
        ;;
      multi)
        FEATURES='[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39]'
        ;;
      all)
        FEATURES='null'
        ;;
    esac

    run_name="UTAE_${feature_set}_fold_${fold}"
    run_dir="$OUTPUT_DIR/$feature_set/fold_$fold"
    mkdir -p "$run_dir"

    echo "=== Running UTAE Table 2 row | feature_set=$feature_set | fold=$fold ==="

    PYTHONPATH="$PWD:$PWD/src" WANDB_MODE=online python src/train.py \
      --config cfgs/UTAE/all_features_dumb_doy.yaml \
      --trainer cfgs/trainer_single_gpu.yaml \
      --data cfgs/data_multitemporal_full_features.yaml \
      --trainer.default_root_dir "$run_dir" \
      --data.data_dir "$DATA_DIR" \
      --data.data_fold_id "$fold" \
      --data.features_to_keep "$FEATURES" \
      --data.return_doy false \
      --trainer.logger.init_args.group "$GROUP_NAME" \
      --trainer.logger.init_args.name "$run_name" \
      --model.init_args.encoder_weights none \
      --optimizer.init_args.lr 0.01 \
      --trainer.accelerator cpu \
      --data.return_doy false \
      --trainer.devices 1 \
      --data.num_workers 0 \
      --trainer.max_steps 2 \
      --do_train true \
      --do_test true \
      --do_predict false

  done
done

echo "✅ Completed all 36 runs for the penultimate Table 2 experiment (UTAE, T=5)."