#!/usr/bin/env bash
# 5-fold cross-validation for one challenge on the L4.
#   bash run_cv.sh GLI            # or MEN, PED
#   bash run_cv.sh PED 0 1 2      # only specific folds (optional)
#
# Trains each fold (Swin UNETR, L4 config: batch 2, 8 workers, no gradient
# checkpointing), runs full-volume evaluation per fold, then aggregates the
# 5-fold mean +/- std. Each fold resumes from its own latest checkpoint, so the
# script is safe to re-run after an interruption.
set -euo pipefail

CH="${1:?usage: run_cv.sh <GLI|MEN|PED> [folds...]}"; shift || true
FOLDS=("$@"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1 2 3 4)

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$PROJ"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
PY=python

case "$CH" in
  GLI) DATA="dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"; LR=1e-4; WD=1e-5; FG=0.9 ;;
  MEN) DATA="dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train"; LR=1e-4; WD=1e-5; FG=0.9 ;;
  PED) DATA="dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"; LR=5e-5; WD=1e-4; FG=0.95 ;;
  *) echo "unknown challenge: $CH"; exit 1 ;;
esac
SPLIT="splits/${CH}_5fold_split.json"

for FOLD in "${FOLDS[@]}"; do
  OUT="output/${CH}_swin_fold${FOLD}"
  echo "=========================================================="
  echo " $CH — FOLD $FOLD : training"
  echo "=========================================================="
  $PY train_swin_unetr.py --model swin --challenge "$CH" --data_dir "$DATA" \
      --split_path "$SPLIT" --fold "$FOLD" --cache_dir "cache/$CH" --out_dir "$OUT" \
      --epochs 300 --lr "$LR" --weight_decay "$WD" --fg_prob "$FG" --backup_interval 100 \
      --batch_size 2 --num_workers 8 --no_checkpoint

  echo "---- $CH fold $FOLD : full-volume evaluation ----"
  $PY evaluate_fullvol.py --challenge "$CH" --data_dir "$DATA" \
      --split_path "$SPLIT" --fold "$FOLD" --ckpt_path "$OUT/latest_ckpt.pth.tar" \
      --cache_dir "cache/$CH" --out_dir "$OUT"
done

echo "=========================================================="
echo " $CH — aggregating 5-fold results"
echo "=========================================================="
$PY aggregate_cv.py --challenge "$CH"
