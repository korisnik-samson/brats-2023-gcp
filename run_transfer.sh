#!/usr/bin/env bash
# GLI -> PED transfer learning (H4) on the L4.
# Warm-starts a PED model from the GLI fold-0 checkpoint, fine-tunes, then evaluates.
# Compare against the from-scratch PED run (output/PED_swin_fold0).
#   bash run_transfer.sh
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$PROJ"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

PED="dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"
OUT="output/PED_transfer_swin_fold0"

echo "=== PED training, warm-started from GLI ==="
python train_swin_unetr.py --challenge PED --data_dir "$PED" \
  --split_path splits/PED_5fold_split.json --fold 0 --cache_dir cache/PED \
  --out_dir "$OUT" --epochs 300 --lr 5e-5 --weight_decay 1e-4 --fg_prob 0.95 \
  --backup_interval 100 --batch_size 2 --num_workers 8 --no_checkpoint \
  --init_from output/GLI_swin_fold0/latest_ckpt.pth.tar

echo "=== Full-volume evaluation ==="
python evaluate_fullvol.py --challenge PED --data_dir "$PED" \
  --split_path splits/PED_5fold_split.json --fold 0 \
  --ckpt_path "$OUT/latest_ckpt.pth.tar" --cache_dir cache/PED --out_dir "$OUT"
