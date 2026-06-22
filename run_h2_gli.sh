#!/usr/bin/env bash
# H2 on the LARGE dataset: 3D U-Net vs Swin UNETR on GLI (L4).
# Tests whether the architecture ranking found on PED reverses with data scale.
#   bash run_h2_gli.sh
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$PROJ"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

GLI="dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
OUT="output/GLI_unet_fold0"

echo "=== GLI training - 3D U-Net baseline ==="
python train_swin_unetr.py --model unet --challenge GLI --data_dir "$GLI" \
  --split_path splits/GLI_5fold_split.json --fold 0 --cache_dir cache/GLI \
  --out_dir "$OUT" --epochs 300 --lr 1e-4 --weight_decay 1e-5 --fg_prob 0.9 \
  --backup_interval 100 --batch_size 2 --num_workers 8

echo "=== Full-volume evaluation ==="
python evaluate_fullvol.py --challenge GLI --data_dir "$GLI" \
  --split_path splits/GLI_5fold_split.json --fold 0 \
  --ckpt_path "$OUT/latest_ckpt.pth.tar" --cache_dir cache/GLI --out_dir "$OUT"
