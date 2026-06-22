# Running BraTS 2023 on a GCP NVIDIA L4

End-to-end guide for 5-fold cross-validation + ensembling on an L4 (24 GB, Linux).
The Python code is unchanged from the Windows version; only the launchers (`.sh`) and the
config (bigger batch, real data-loader workers, no gradient checkpointing) differ.

## 0. Instance
- Machine type **g2-standard-8** (1× L4, 8 vCPU, 32 GB RAM) or **g2-standard-16** (64 GB RAM
  — better for `--num_workers 8`). Boot image: a Deep Learning VM (CUDA pre-installed) or
  Ubuntu 22.04 + NVIDIA driver. Confirm with `nvidia-smi`.

## 1. Code
```bash
git clone <your-repo> brats-2023 && cd brats-2023      # or scp the project up
```

## 2. Data
The `dataset/` tree is large (tens of GB). Fastest path is via a GCS bucket:
```bash
# from the machine that has the data (one-time upload):
gsutil -m cp -r dataset gs://<your-bucket>/brats-2023/dataset
# on the L4 instance:
gsutil -m cp -r gs://<your-bucket>/brats-2023/dataset .
```
Alternative (slower, direct): `gcloud compute scp --recurse dataset <instance>:~/brats-2023/`.
The small `splits/*.json` are in the repo already — **do not regenerate them** (reproducibility).
Re-apply the MEN V4 patch if you re-download MEN (see methodology); the committed splits assume it.

## 3. Environment
```bash
bash setup.sh                       # venv at ~/brats-env + CUDA torch + deps + GPU check
source ~/brats-env/bin/activate
```

## 4. Fit-check (one short run)
Confirm the L4 config fits before committing to long runs:
```bash
python train_swin_unetr.py --challenge PED \
  --data_dir dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData \
  --split_path splits/PED_5fold_split.json --fold 0 --cache_dir cache/PED \
  --out_dir output/_fit --epochs 1 --batch_size 2 --num_workers 8 --no_checkpoint
```
If it OOMs (it should not on 24 GB), drop `--no_checkpoint` first, then `--batch_size 1`.
If it fits with headroom, you can push `--batch_size 4` (keep checkpointing) for more throughput.

## 5. Five-fold cross-validation
Each call trains folds 0–4, evaluates each on its held-out fold, and aggregates:
```bash
bash run_cv.sh PED        # ~hours
bash run_cv.sh MEN        # longer
bash run_cv.sh GLI        # longest
```
`run_cv.sh` uses `--batch_size 2 --num_workers 8 --no_checkpoint`. It resumes each fold from
its checkpoint, so it is safe to stop/restart. Re-aggregate any time:
```bash
python aggregate_cv.py --challenge GLI       # pooled mean±std + per-fold means
```
This 5-fold mean±std is the robust number that replaces the single-fold result in the thesis
(directly addressing the §5.8 limitation).

## 6. Ensemble (for the official leaderboard number)
Average the 5 fold models. On **labelled** held-out data it prints metrics; on the official
**unlabelled** ValidationData it writes label NIfTIs for submission:
```bash
# submission NIfTIs from the 5-model ensemble:
python evaluate_ensemble.py --challenge GLI \
  --data_dir dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/<inner subject dir> \
  --ckpt_paths "output/GLI_swin_fold*/latest_ckpt.pth.tar" --save_nifti --out_dir submission/GLI
```
Avoid scoring the all-fold ensemble on a training fold (leakage) — use ValidationData or a
fully held-out split for a clean labelled ensemble number.

## 7. Remaining fold-0 experiments (also faster here)
- U-Net on GLI (H2 at scale): `run_h2_gli.ps1` logic →
  `python train_swin_unetr.py --model unet --challenge GLI ... --batch_size 2 --num_workers 8`
- GLI→PED transfer: add `--init_from output/GLI_swin_fold0/latest_ckpt.pth.tar`.
- Ablation: re-run evaluate_fullvol with different `--min_cc` / thresholds (no retraining).

## Config summary (L4 vs the 12 GB Windows box)
| Setting | RTX 4070 (12 GB) | L4 (24 GB) |
|---|---|---|
| `--batch_size` | 1 | 2 (up to 4) |
| `--num_workers` | 0 (Windows spawn) | 8 (Linux fork) |
| gradient checkpointing | on (required) | `--no_checkpoint` (faster) |
| scope | fold 0 only | 5-fold CV + ensemble |
