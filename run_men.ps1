# BraTS 2023 Swin UNETR — MEN training + validation (single GPU, sequential)
# Run from a STANDALONE terminal (not inside PyCharm) with heavy apps closed.
#
#   powershell -ExecutionPolicy Bypass -File run_men.ps1
#
# Resumes automatically from output/MEN_swin_fold0/latest_ckpt.pth.tar if interrupted.

$ErrorActionPreference = "Continue"
$py = "C:\Users\sammi\anaconda3\envs\machine-learning-env\python.exe"
Set-Location "C:\Users\sammi\Desktop\projects\brats-2023"

# Live, unbuffered output; quiet deprecation warnings. Do NOT set max_split_size_mb
# (it fragments VRAM for Swin and causes OOM — the default allocator works).
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS  = "ignore"

$MEN_DIR = "dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train"

Write-Host "`n=== [1/2] MEN training (resumes from latest_ckpt) ===" -ForegroundColor Cyan
& $py train_swin_unetr.py --challenge MEN --data_dir $MEN_DIR `
    --split_path splits/MEN_5fold_split.json --fold 0 --cache_dir cache/MEN `
    --out_dir output/MEN_swin_fold0 --epochs 300 --lr 1e-4 --weight_decay 1e-5 `
    --fg_prob 0.9 --backup_interval 50
$menTrain = $LASTEXITCODE

if ($menTrain -eq 0) {
    Write-Host "`n=== [2/2] MEN validation ===" -ForegroundColor Cyan
    & $py validate_swin_unetr.py --challenge MEN --data_dir $MEN_DIR `
        --split_path splits/MEN_5fold_split.json --fold 0 `
        --ckpt_path output/MEN_swin_fold0/latest_ckpt.pth.tar `
        --cache_dir cache/MEN --out_dir output/MEN_swin_fold0
} else {
    Write-Host "MEN training exited with code $menTrain - skipping MEN validation." -ForegroundColor Yellow
}

Write-Host "`n=== MEN pipeline complete ===" -ForegroundColor Green
