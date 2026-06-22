#!/usr/bin/env bash
# Download + extract + arrange the BraTS 2023 data from Synapse, on the VM.
#
# The Synapse personal access token is read from the environment — it is NEVER stored
# in this file. Set it just for the session before running:
#
#   export SYNAPSE_AUTH_TOKEN='<your synapse token>'
#   bash download_data.sh
#
# Downloads only what the pipeline needs (GLI/MEN/PED training + official validation +
# the MEN V4 patch), extracts everything, symlinks the subject folders to the exact
# paths the launchers expect (regardless of each archive's internal nesting), and applies
# the MEN V4 patch. The large synthesis / fastlane bundles and the .xlsx metadata are
# skipped (not used by the pipeline).
set -euo pipefail
: "${SYNAPSE_AUTH_TOKEN:?Set your token first:  export SYNAPSE_AUTH_TOKEN='...'}"

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$PROJ"
command -v unzip >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y unzip; }
pip install --quiet --upgrade synapseclient

mkdir -p downloads staging dataset
echo "=== Downloading from Synapse (token read from \$SYNAPSE_AUTH_TOKEN) ==="
cd downloads
synapse get syn51514132   # ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip
synapse get syn51514110   # ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip
synapse get syn51514055   # ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData.zip  (current = v3)
synapse get syn51930467   # ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData.zip
synapse get syn51718026   # BraTS-MEN-TRAIN-FIX-V4.zip
synapse get syn51615054   # ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData.zip
synapse get syn51929861   # ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData.zip
cd ..

echo "=== Extracting ==="
for z in downloads/*.zip; do
  base="$(basename "${z%.zip}")"
  echo "  $z"
  unzip -q -o "$z" -d "staging/$base"
done

# Link the directory that DIRECTLY contains BraTS-<CH>-* subject folders to the
# canonical path the launchers/splits expect (works whatever the zip's nesting).
canon () {  # $1=challenge  $2=staging subdir  $3=canonical target path
  local subj_parent
  subj_parent="$(dirname "$(find "staging/$2" -type d -name "BraTS-$1-*" -print -quit)")"
  [ -d "$subj_parent" ] || { echo "!! no BraTS-$1-* found under staging/$2"; return 1; }
  mkdir -p "$(dirname "$3")"
  ln -sfn "$(realpath "$subj_parent")" "$3"
  echo "  $3  ->  $subj_parent  ($(find "$subj_parent" -maxdepth 1 -type d -name "BraTS-$1-*" | wc -l) subjects)"
}
echo "=== Arranging into dataset/ ==="
canon GLI ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData   "dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
canon GLI ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData "dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
canon PED ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData   "dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"
canon PED ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData "dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData"
canon MEN ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData   "dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train"
canon MEN ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData "dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData"

echo "=== Applying MEN V4 patch (corrects modality images for ~48 subjects) ==="
V4_PARENT="$(dirname "$(find staging/BraTS-MEN-TRAIN-FIX-V4 -type d -name 'BraTS-MEN-*' -print -quit)")"
MEN_TRAIN="dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train"
n=0
for d in "$V4_PARENT"/BraTS-MEN-*; do
  s="$(basename "$d")"
  [ -d "$MEN_TRAIN/$s" ] && cp -f "$d"/*.nii.gz "$MEN_TRAIN/$s/" && n=$((n+1))
done
echo "  patched $n subjects"

echo
echo "=== Done. Sanity check ==="
for CH in GLI MEN PED; do
  case $CH in
    GLI) P="dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" ;;
    MEN) P="dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train" ;;
    PED) P="dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData" ;;
  esac
  echo "  $CH train subjects: $(find "$P" -maxdepth 1 -type d -name "BraTS-$CH-*" 2>/dev/null | wc -l)"
done
echo "You can delete downloads/ to reclaim space once everything looks right."
