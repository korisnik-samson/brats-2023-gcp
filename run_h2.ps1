# BraTS 2023 - H2 architecture comparison: 3D U-Net vs Swin UNETR (on PED)
# Trains the 3D U-Net baseline under conditions IDENTICAL to the from-scratch PED
# Swin run (output/PED_swin_fold0): same fold, patch, batch, LR schedule, loss,
# augmentation and 300-epoch budget; only the architecture differs.
# Compare its validation to output/PED_swin_fold0 to isolate the architecture effect.
# Run from a STANDALONE terminal with heavy apps closed.
#
#   powershell -ExecutionPolicy Bypass -File run_h2.ps1
#
# PED chosen for a fast ~2 h comparison; the PED disk cache is reused (cached
# volumes are model-independent, so there is no re-preprocessing).

$ErrorActionPreference = "Continue"
$py = "C:\Users\sammi\anaconda3\envs\machine-learning-env\python.exe"
Set-Location "C:\Users\sammi\Desktop\projects\brats-2023"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS  = "ignore"

$PED_DIR = "dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"

Write-Host "`n=== [1/2] PED training - 3D U-Net baseline (resumes if interrupted) ===" -ForegroundColor Cyan
& $py train_swin_unetr.py --model unet --challenge PED --data_dir $PED_DIR `
    --split_path splits/PED_5fold_split.json --fold 0 --cache_dir cache/PED `
    --out_dir output/PED_unet_fold0 --epochs 300 --lr 5e-5 --weight_decay 1e-4 `
    --fg_prob 0.95 --backup_interval 50
$tr = $LASTEXITCODE

if ($tr -eq 0) {
    Write-Host "`n=== [2/2] Validation (cropped voxel) ===" -ForegroundColor Cyan
    & $py validate_swin_unetr.py --challenge PED --data_dir $PED_DIR `
        --split_path splits/PED_5fold_split.json --fold 0 `
        --ckpt_path output/PED_unet_fold0/latest_ckpt.pth.tar `
        --cache_dir cache/PED --out_dir output/PED_unet_fold0
} else {
    Write-Host "U-Net training exited with code $tr - skipping validation." -ForegroundColor Yellow
}
Write-Host "`n=== H2 (U-Net) run complete ===" -ForegroundColor Green
