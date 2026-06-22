#!/usr/bin/env bash
# H2 architecture comparison on PED: 3D U-Net vs Swin UNETR (L4).
# Identical conditions to the from-scratch PED Swin run; only the architecture differs.
#   bash run_h2.sh
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$PROJ"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

PED="dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"
OUT="output/PED_unet_fold0"

echo "=== PED training - 3D U-Net baseline ==="
python train_swin_unetr.py --model unet --challenge PED --data_dir "$PED" \
  --split_path splits/PED_5fold_split.json --fold 0 --cache_dir cache/PED \
  --out_dir "$OUT" --epochs 300 --lr 5e-5 --weight_decay 1e-4 --fg_prob 0.95 \
  --backup_interval 100 --batch_size 2 --num_workers 8

echo "=== Full-volume evaluation ==="
python evaluate_fullvol.py --challenge PED --data_dir "$PED" \
  --split_path splits/PED_5fold_split.json --fold 0 \
  --ckpt_path "$OUT/latest_ckpt.pth.tar" --cache_dir cache/PED --out_dir "$OUT"
