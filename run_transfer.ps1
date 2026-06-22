# BraTS 2023 — GLI -> PED transfer learning (H4)
# Warm-starts a PED model from the GLI-trained checkpoint, then fine-tunes on PED.
# Compare its results against the from-scratch PED run (output/PED_swin_fold0).
# Run from a STANDALONE terminal with heavy apps closed.
#
#   powershell -ExecutionPolicy Bypass -File run_transfer.ps1
#
# For the freeze-encoder variant (fine-tune decoder only), add --freeze_encoder
# to the training line below.

$ErrorActionPreference = "Continue"
$py = "C:\Users\sammi\anaconda3\envs\machine-learning-env\python.exe"
Set-Location "C:\Users\sammi\Desktop\projects\brats-2023"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS  = "ignore"

$PED_DIR = "dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"

Write-Host "`n=== [1/2] PED training, warm-started from GLI (resumes if interrupted) ===" -ForegroundColor Cyan
& $py train_swin_unetr.py --challenge PED --data_dir $PED_DIR `
    --split_path splits/PED_5fold_split.json --fold 0 --cache_dir cache/PED `
    --out_dir output/PED_transfer_swin_fold0 --epochs 300 --lr 5e-5 --weight_decay 1e-4 `
    --fg_prob 0.95 --backup_interval 50 `
    --init_from output/GLI_swin_fold0/latest_ckpt.pth.tar
$tr = $LASTEXITCODE

if ($tr -eq 0) {
    Write-Host "`n=== [2/2] Validation (cropped voxel) ===" -ForegroundColor Cyan
    & $py validate_swin_unetr.py --challenge PED --data_dir $PED_DIR `
        --split_path splits/PED_5fold_split.json --fold 0 `
        --ckpt_path output/PED_transfer_swin_fold0/latest_ckpt.pth.tar `
        --cache_dir cache/PED --out_dir output/PED_transfer_swin_fold0
} else {
    Write-Host "Transfer training exited with code $tr - skipping validation." -ForegroundColor Yellow
}
Write-Host "`n=== Transfer run complete ===" -ForegroundColor Green
