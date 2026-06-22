# BraTS 2023 - H2 on the LARGE dataset: 3D U-Net vs Swin UNETR (on GLI)
# Trains the 3D U-Net on GLI under conditions IDENTICAL to the GLI Swin run
# (output/GLI_swin_fold0): same fold, patch, batch, LR schedule, loss, augmentation
# and 300-epoch budget; only the architecture differs. Tests whether the
# architecture ranking found on PED reverses with data scale.
# Run from a STANDALONE terminal with heavy apps closed. ~20-24 h, resumable.
#
#   powershell -ExecutionPolicy Bypass -File run_h2_gli.ps1
#
# Reuses the GLI disk cache (cached volumes are model-independent).

$ErrorActionPreference = "Continue"
$py = "C:\Users\sammi\anaconda3\envs\machine-learning-env\python.exe"
Set-Location "C:\Users\sammi\Desktop\projects\brats-2023"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS  = "ignore"

$GLI_DIR = "dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

Write-Host "`n=== [1/2] GLI training - 3D U-Net baseline (resumes if interrupted) ===" -ForegroundColor Cyan
& $py train_swin_unetr.py --model unet --challenge GLI --data_dir $GLI_DIR `
    --split_path splits/GLI_5fold_split.json --fold 0 --cache_dir cache/GLI `
    --out_dir output/GLI_unet_fold0 --epochs 300 --lr 1e-4 --weight_decay 1e-5 `
    --fg_prob 0.9 --backup_interval 50
$tr = $LASTEXITCODE

if ($tr -eq 0) {
    Write-Host "`n=== [2/2] Validation (cropped voxel) ===" -ForegroundColor Cyan
    & $py validate_swin_unetr.py --challenge GLI --data_dir $GLI_DIR `
        --split_path splits/GLI_5fold_split.json --fold 0 `
        --ckpt_path output/GLI_unet_fold0/latest_ckpt.pth.tar `
        --cache_dir cache/GLI --out_dir output/GLI_unet_fold0
} else {
    Write-Host "U-Net (GLI) training exited with code $tr - skipping validation." -ForegroundColor Yellow
}
Write-Host "`n=== H2-GLI (U-Net) run complete ===" -ForegroundColor Green
