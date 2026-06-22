# BraTS 2023 Swin UNETR — full pipeline (single GPU, sequential)
# PED train (resumes from latest checkpoint) -> PED validate -> GLI train -> GLI validate.
# Run from a STANDALONE terminal (not inside PyCharm) with heavy apps closed.
#
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
#
# Durable artifacts are written by the Python scripts themselves:
#   output/<run>/training_loss.csv, latest_ckpt.pth.tar, backup_ckpts/, val_metrics_*.csv

$ErrorActionPreference = "Continue"
$py = "C:\Users\sammi\anaconda3\envs\machine-learning-env\python.exe"
Set-Location "C:\Users\sammi\Desktop\projects\brats-2023"

# Live, unbuffered output and quiet deprecation warnings (those warnings going to
# stderr otherwise make PowerShell print noisy NativeCommandError records).
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb here — it fragments
# VRAM for Swin's large activations and causes OOM. The default allocator works.
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS  = "ignore"

$PED_DIR = "dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"
$GLI_DIR = "dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

Write-Host "`n=== [1/4] PED training (resumes from latest_ckpt) ===" -ForegroundColor Cyan
& $py train_swin_unetr.py --challenge PED --data_dir $PED_DIR `
    --split_path splits/PED_5fold_split.json --fold 0 --cache_dir cache/PED `
    --out_dir output/PED_swin_fold0 --epochs 300 --lr 5e-5 --weight_decay 1e-4 `
    --fg_prob 0.95 --backup_interval 50
$pedTrain = $LASTEXITCODE

if ($pedTrain -eq 0) {
    Write-Host "`n=== [2/4] PED validation ===" -ForegroundColor Cyan
    & $py validate_swin_unetr.py --challenge PED --data_dir $PED_DIR `
        --split_path splits/PED_5fold_split.json --fold 0 `
        --ckpt_path output/PED_swin_fold0/latest_ckpt.pth.tar `
        --cache_dir cache/PED --out_dir output/PED_swin_fold0
} else {
    Write-Host "PED training exited with code $pedTrain - skipping PED validation." -ForegroundColor Yellow
}

Write-Host "`n=== [3/4] GLI training ===" -ForegroundColor Cyan
& $py train_swin_unetr.py --challenge GLI --data_dir $GLI_DIR `
    --split_path splits/GLI_5fold_split.json --fold 0 --cache_dir cache/GLI `
    --out_dir output/GLI_swin_fold0 --epochs 300 --lr 1e-4 --weight_decay 1e-5 `
    --fg_prob 0.9 --backup_interval 50
$gliTrain = $LASTEXITCODE

if ($gliTrain -eq 0) {
    Write-Host "`n=== [4/4] GLI validation ===" -ForegroundColor Cyan
    & $py validate_swin_unetr.py --challenge GLI --data_dir $GLI_DIR `
        --split_path splits/GLI_5fold_split.json --fold 0 `
        --ckpt_path output/GLI_swin_fold0/latest_ckpt.pth.tar `
        --cache_dir cache/GLI --out_dir output/GLI_swin_fold0
} else {
    Write-Host "GLI training exited with code $gliTrain - skipping GLI validation." -ForegroundColor Yellow
}

Write-Host "`n=== Pipeline complete ===" -ForegroundColor Green
