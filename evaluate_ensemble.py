"""
Ensemble inference / evaluation: average the probability maps of several fold models.

For each subject the in-distribution crop is run through every checkpoint, the sigmoid
probabilities are averaged, and the averaged map is converted to a prediction (uncropped
to native space, dust-removed) exactly as in evaluate_fullvol.py. If a ground-truth
segmentation is present the per-region voxel / lesion-wise metrics are reported; otherwise
(e.g. the official ValidationData) a label NIfTI is written for leaderboard submission.

Examples
  # Ensemble all 5 GLI folds, score on a held-out labelled set:
  python evaluate_ensemble.py --challenge GLI --data_dir <dir-with-seg> \
      --ckpt_paths output/GLI_swin_fold*/latest_ckpt.pth.tar --out_dir output/GLI_ensemble

  # Ensemble + write NIfTI predictions for the official (unlabelled) ValidationData:
  python evaluate_ensemble.py --challenge GLI \
      --data_dir dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/... \
      --ckpt_paths output/GLI_swin_fold*/latest_ckpt.pth.tar --save_nifti --out_dir submission/GLI

Note on leakage: do NOT ensemble all folds and score on a fold whose validation subjects
were in the other folds' training data. For a clean labelled ensemble number use data none
of the models trained on (the official ValidationData, or a fully held-out split).
"""
import os
import csv
import json
import glob
import argparse

import numpy as np
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference

from loader.brats_dataset import MODALITIES
from utils.general_utils import overlapping_probs_to_preds, disjoint_to_overlapping, one_hot_channels_to_three_labels
from evaluate_fullvol import (
    preprocess_cropped, uncrop, remove_dust, gt_regions,
    voxel_dice, voxel_hd95, lesionwise, REGION_NAMES,
)


def main():
    ap = argparse.ArgumentParser(description="Ensemble inference / evaluation across fold checkpoints")
    ap.add_argument("--challenge", default="GLI", choices=["GLI", "MEN", "PED"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt_paths", nargs="+", required=True, help="Fold checkpoints (globs allowed)")
    ap.add_argument("--split_path", default=None, help="If set with --fold, evaluate that fold's val list")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out_dir", default="./output/ensemble")
    ap.add_argument("--save_nifti", action="store_true", help="Write label NIfTIs (for unlabelled submission data)")
    ap.add_argument("--crop", type=int, nargs=3, default=[192, 192, 128])
    ap.add_argument("--roi", type=int, default=128)
    ap.add_argument("--sw_batch_size", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--t_wt", type=float, default=0.45)
    ap.add_argument("--t_tc", type=float, default=0.40)
    ap.add_argument("--t_et", type=float, default=0.45)
    ap.add_argument("--min_cc", type=int, default=50)
    ap.add_argument("--dilation", type=int, default=2)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    roi = (args.roi,) * 3
    crop = tuple(args.crop)
    thr = (args.t_wt, args.t_tc, args.t_et)

    # resolve checkpoint globs
    ckpts = []
    for pat in args.ckpt_paths:
        ckpts += sorted(glob.glob(pat)) or [pat]
    ckpts = [c for c in ckpts if os.path.exists(c)]
    if not ckpts:
        raise SystemExit("No checkpoints found.")
    print(f"Ensembling {len(ckpts)} model(s):")
    models = []
    for c in ckpts:
        print("  -", c)
        ck = torch.load(c, map_location=device, weights_only=False)
        m = ck["model"]; m.load_state_dict(ck["model_sd"]); models.append(m.to(device).eval())

    # subject list
    if args.split_path and args.fold is not None:
        subjects = json.load(open(args.split_path))["folds"][str(args.fold)]["val"]
    else:
        subjects = sorted(d for d in os.listdir(args.data_dir)
                          if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith(f"BraTS-{args.challenge}"))
    print(f"Subjects: {len(subjects)} | save_nifti={args.save_nifti}\n")
    os.makedirs(args.out_dir, exist_ok=True)

    acc = {f"{m}_{r}": [] for m in ("vdsc", "vhd", "ldsc", "lhd") for r in REGION_NAMES}
    rows, scored = [], 0

    with torch.no_grad():
        for i, name in enumerate(subjects):
            sd = os.path.join(args.data_dir, name)
            x, coords, orig_shape = preprocess_cropped(sd, name, crop)
            x = x.to(device)

            prob = None
            for m in models:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = sliding_window_inference(x, roi, args.sw_batch_size, m,
                                                      overlap=args.overlap, mode="gaussian")
                p = torch.sigmoid(logits.float())
                prob = p if prob is None else prob + p
            prob /= len(models)                                   # averaged probability map

            preds_disjoint = overlapping_probs_to_preds(prob, t1=thr[0], t2=thr[1], t3=thr[2]).float()
            pred_ov = uncrop(disjoint_to_overlapping(preds_disjoint)[0].cpu().numpy().astype(bool),
                             coords, orig_shape, crop)
            pred_ov = remove_dust(pred_ov, args.min_cc)

            seg_path = os.path.join(sd, f"{name}-seg.nii.gz")
            if os.path.exists(seg_path):
                seg = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.int16)
                gts = gt_regions(seg)
                row = {"subject": name}
                for c, r in enumerate(REGION_NAMES):
                    p, g = pred_ov[c], gts[c]
                    vd = voxel_dice(p, g); vh = voxel_hd95(p, g)
                    ld, lh = lesionwise(p, g, dilation=args.dilation)
                    acc[f"vdsc_{r}"].append(vd); acc[f"ldsc_{r}"].append(ld); acc[f"lhd_{r}"].append(lh)
                    if vh is not None:
                        acc[f"vhd_{r}"].append(vh)
                    row.update({f"vdsc_{r}": round(vd, 4), f"ldsc_{r}": round(ld, 4)})
                rows.append(row); scored += 1
                print(f"[{i+1}/{len(subjects)}] {name} | " +
                      " ".join(f"{r} v{row[f'vdsc_{r}']:.2f}/L{row[f'ldsc_{r}']:.2f}" for r in REGION_NAMES))

            if args.save_nifti:
                disj = disjoint_to_overlapping  # noqa  (kept for clarity)
                # rebuild disjoint one-hot in native space, then a single label map
                dj = uncrop(preds_disjoint[0].cpu().numpy().astype(bool), coords, orig_shape, crop)
                label = one_hot_channels_to_three_labels(dj).astype(np.uint8)
                affine = nib.load(os.path.join(sd, f"{name}-{MODALITIES[0]}.nii.gz")).affine
                nib.save(nib.Nifti1Image(label, affine), os.path.join(args.out_dir, f"{name}.nii.gz"))

    if scored:
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        with open(os.path.join(args.out_dir, "ensemble_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subject"] + [f"{m}_{r}" for r in REGION_NAMES for m in ("vdsc", "ldsc")])
            w.writeheader(); w.writerows(rows)
        print("\n=== ENSEMBLE (%d models, %d subjects) ===" % (len(models), scored))
        print(f"{'Region':<7}{'voxel DSC':<12}{'lesion DSC':<12}")
        for r in REGION_NAMES:
            print(f"{r:<7}{mean(acc[f'vdsc_{r}']):<12.4f}{mean(acc[f'ldsc_{r}']):<12.4f}")
        print(f"Mean voxel DSC : {mean([mean(acc[f'vdsc_{r}']) for r in REGION_NAMES]):.4f}")
        print(f"Mean lesion DSC: {mean([mean(acc[f'ldsc_{r}']) for r in REGION_NAMES]):.4f}")
    if args.save_nifti:
        print(f"\nNIfTI predictions written to: {args.out_dir}")


if __name__ == "__main__":
    main()
