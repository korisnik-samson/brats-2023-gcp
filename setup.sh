#!/usr/bin/env bash
# One-time environment setup on a GCP L4 (Ubuntu/Debian) instance.
#   bash setup.sh
# Creates a venv at ~/brats-env, installs a CUDA PyTorch build + the project deps,
# and verifies the GPU is visible.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="${HOME}/brats-env"

echo "=== Creating virtual environment at ${ENV} ==="
python3 -m venv "${ENV}"
# shellcheck disable=SC1091
source "${ENV}/bin/activate"
pip install --upgrade pip wheel

echo "=== Installing PyTorch (CUDA 12.4 build for the L4 / Ada) ==="
# If the L4 instance ships a different CUDA, switch cu124 -> cu121 or cu126.
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing project requirements ==="
pip install -r "${PROJ}/requirements.txt"

echo "=== Verifying ==="
python - <<'PY'
import torch, monai
print("torch :", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("monai :", monai.__version__)
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo
echo "Done. Activate with:  source ${ENV}/bin/activate"
echo "Then run a single fold to fit-check, e.g.:"
echo "  python train_swin_unetr.py --challenge PED --data_dir <PED inner dir> \\"
echo "    --split_path splits/PED_5fold_split.json --fold 0 --cache_dir cache/PED \\"
echo "    --out_dir output/PED_swin_fold0 --epochs 1 --batch_size 2 --num_workers 8 --no_checkpoint"
