"""
Sliding-Window Validation for BraTS 2023 Swin UNETR
===================================================
Evaluates a patch-trained Swin UNETR checkpoint on a fold's validation split
using MONAI sliding-window inference (the patch-trained model must NOT be fed a
full volume directly). Reports per-region Dice and HD95 (WT / TC / ET) following
the BraTS evaluation conventions (methodology section 9).

Edge cases (methodology 9.1):
  * Dice: empty pred vs empty GT  -> 1.0 (both agree there is no tumour)
          empty pred vs non-empty -> 0.0
  * HD95: undefined when either pred or GT for a region is empty -> the subject
          is skipped for that region's HD95 and the skip is counted.

Usage:
    python validate_swin_unetr.py --challenge PED \
        --data_dir dataset/.../ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData \
        --split_path splits/PED_5fold_split.json --fold 0 \
        --ckpt_path output/PED_swin_fold0/latest_ckpt.pth.tar \
        --cache_dir cache/PED --out_dir output/PED_swin_fold0
"""

import os
import csv
import argparse

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance

from loader.brats_dataset import BraTSDataset
from utils.general_utils import (
    seg_to_one_hot_channels, disjoint_to_overlapping, overlapping_probs_to_preds,
)

REGION_NAMES = ["WT", "TC", "ET"]


def region_dice(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> float:
    """Binary Dice for one region with BraTS empty-region conventions."""
    p_sum = float(pred_bin.sum())
    g_sum = float(gt_bin.sum())
    if g_sum == 0 and p_sum == 0:
        return 1.0
    if g_sum == 0 and p_sum > 0:
        return 0.0
    inter = float((pred_bin * gt_bin).sum())
    return 2.0 * inter / (p_sum + g_sum)


def region_hd95(pred_bin: torch.Tensor, gt_bin: torch.Tensor):
    """HD95 for one region, or None when either mask is empty (undefined)."""
    if float(pred_bin.sum()) == 0 or float(gt_bin.sum()) == 0:
        return None
    # compute_hausdorff_distance expects (B, C, H, W, D) one-hot tensors.
    p = pred_bin[None, None].float()
    g = gt_bin[None, None].float()
    hd = compute_hausdorff_distance(p, g, include_background=True, percentile=95)
    return float(hd.item())


def main():
    parser = argparse.ArgumentParser(description="BraTS 2023 Swin UNETR sliding-window validation")
    parser.add_argument("--challenge", type=str, default="PED", choices=["GLI", "MEN", "PED"])
    parser.add_argument("--data_dir", type=str, required=True, help="Inner challenge training-data folder")
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to latest_ckpt.pth.tar")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./output", help="Where to write the metrics CSV")
    parser.add_argument("--crop", type=int, nargs=3, default=[192, 192, 128], help="Full-volume crop (H W D)")
    parser.add_argument("--roi", type=int, default=128, help="Sliding-window ROI (cubic)")
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--t_wt", type=float, default=0.45, help="WT threshold")
    parser.add_argument("--t_tc", type=float, default=0.40, help="TC threshold")
    parser.add_argument("--t_et", type=float, default=0.45, help="ET threshold")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Validate only the first N subjects (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    roi_size = (args.roi, args.roi, args.roi)
    crop_target = tuple(args.crop)

    # ── Load checkpoint (our checkpoints store full objects → weights_only=False) ──
    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model = ckpt["model"]
    model.load_state_dict(ckpt["model_sd"])
    model = model.to(device).eval()
    training_regions = ckpt.get("training_regions", "overlapping")
    print(f"Loaded epoch {ckpt.get('epoch')} | challenge={ckpt.get('challenge')} | "
          f"fold={ckpt.get('fold')} | training_regions={training_regions}")
    if training_regions != "overlapping":
        raise ValueError("This validator assumes a model trained on overlapping regions (WT/TC/ET).")

    # ── Validation dataset: same preprocessing as training, no augmentation ──
    ds = BraTSDataset(
        args.data_dir, challenge=args.challenge, mode="val", fold=args.fold,
        split_path=args.split_path, transform=None, cache_dir=args.cache_dir,
        crop_target=crop_target,
    )
    n_subjects = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"Validating on {n_subjects} subjects (fold {args.fold} val split).\n")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"val_metrics_fold{args.fold}_epoch{ckpt.get('epoch')}.csv")
    rows = []

    dice_acc = {r: [] for r in REGION_NAMES}
    hd_acc = {r: [] for r in REGION_NAMES}
    hd_skips = {r: 0 for r in REGION_NAMES}
    thresholds = (args.t_wt, args.t_tc, args.t_et)

    with torch.no_grad():
        for idx in range(n_subjects):
            name, imgs, seg = ds[idx]               # imgs: list of 4×(1,H,W,D); seg: (1,H,W,D)
            x_in = torch.cat(imgs, dim=0).unsqueeze(0).to(device)   # (1,4,H,W,D)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = sliding_window_inference(
                    x_in, roi_size, args.sw_batch_size, model,
                    overlap=args.overlap, mode="gaussian",
                )
            probs = torch.sigmoid(logits.float())   # (1,3,H,W,D) overlapping WT/TC/ET

            # Predicted disjoint labels (with nesting logic) → overlapping masks.
            preds_disjoint = overlapping_probs_to_preds(
                probs, t1=thresholds[0], t2=thresholds[1], t3=thresholds[2]
            ).float()                                 # (1,3,H,W,D) one-hot NCR/ED/ET on CPU
            preds_overlap = disjoint_to_overlapping(preds_disjoint)[0]   # (3,H,W,D)

            # Ground-truth overlapping masks.
            seg_b = seg.unsqueeze(0)                  # (1,1,H,W,D)
            gt_overlap = disjoint_to_overlapping(seg_to_one_hot_channels(seg_b))[0].cpu()

            row = {"subject": name}
            for c, r in enumerate(REGION_NAMES):
                d = region_dice(preds_overlap[c], gt_overlap[c])
                h = region_hd95(preds_overlap[c], gt_overlap[c])
                dice_acc[r].append(d)
                row[f"dice_{r}"] = round(d, 4)
                if h is None:
                    hd_skips[r] += 1
                    row[f"hd95_{r}"] = ""
                else:
                    hd_acc[r].append(h)
                    row[f"hd95_{r}"] = round(h, 3)
            rows.append(row)
            print(f"[{idx+1:>3}/{len(ds)}] {name} | "
                  + " ".join(f"{r} {row[f'dice_{r}']:.3f}" for r in REGION_NAMES))

    # ── Write per-subject CSV ──
    fieldnames = ["subject"] + [f"dice_{r}" for r in REGION_NAMES] + [f"hd95_{r}" for r in REGION_NAMES]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──
    print("\n" + "=" * 56)
    print(f"VALIDATION SUMMARY — {args.challenge} fold {args.fold} (epoch {ckpt.get('epoch')})")
    print("=" * 56)
    print(f"{'Region':<6}{'Dice (mean±std)':<22}{'HD95 (mean, n)':<22}")
    mean_dice = []
    for r in REGION_NAMES:
        d = np.array(dice_acc[r])
        mean_dice.append(d.mean())
        if hd_acc[r]:
            hd = np.array(hd_acc[r])
            hd_str = f"{hd.mean():.2f} (n={len(hd)}, skip={hd_skips[r]})"
        else:
            hd_str = f"— (all {hd_skips[r]} empty)"
        print(f"{r:<6}{d.mean():.4f} ± {d.std():.4f}     {hd_str}")
    print("-" * 56)
    print(f"Mean Dice (WT,TC,ET avg): {np.mean(mean_dice):.4f}")
    print(f"\nPer-subject metrics written to: {csv_path}")


if __name__ == "__main__":
    main()
