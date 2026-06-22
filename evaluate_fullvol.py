"""
Full-volume + lesion-wise evaluation for BraTS 2023 Swin UNETR
=============================================================
Addresses the limitation of `validate_swin_unetr.py`, which scores in the cropped
192x192x128 space with voxel metrics. Here we evaluate in the **original full-resolution
volume** and report BOTH:

  * voxel-level  DSC / HD95  per region (WT, TC, ET), and
  * lesion-wise  DSC / HD95  per region — connected-component matching that penalises
    false-positive and false-negative *lesions*, following the BraTS 2023 definition.

Preprocessing reuses the training-time per-modality normalisation (`_znorm`) so the input
distribution matches training; the model is applied with sliding-window inference over the
full volume (no cropping), so predictions live in native image space and are compared
directly with the full ground-truth segmentation.

Lesion-wise algorithm (documented; our implementation, not the official `panoptica`):
  for each region, label GT and prediction connected components (26-connectivity, cc3d);
  ignore components smaller than `min_lesion_vox`. For each GT lesion, take the union of
  predicted voxels in components overlapping it and score DSC/HD95 against that GT lesion;
  a GT lesion with no overlapping prediction is a false negative (DSC 0, HD95 = penalty);
  a predicted lesion overlapping no GT lesion is a false positive (DSC 0, HD95 = penalty).
  The region score is the mean over {matched GT lesions + FN + FP}. An empty/empty region
  scores DSC 1, HD95 0. `--hd95_penalty` defaults to 374 mm (the BraTS convention).

Usage:
    python evaluate_fullvol.py --challenge PED \
        --data_dir dataset/.../<inner subject folder> \
        --split_path splits/PED_5fold_split.json --fold 0 \
        --ckpt_path output/PED_swin_fold0/latest_ckpt.pth.tar \
        --out_dir output/PED_swin_fold0
"""

import os
import gc
import csv
import json
import argparse

import numpy as np
import torch
import nibabel as nib
import cc3d
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance

from loader.brats_dataset import _znorm, _compute_crop_coords, _pad_to_shape, MODALITIES
from utils.general_utils import overlapping_probs_to_preds, disjoint_to_overlapping

REGION_NAMES = ["WT", "TC", "ET"]


# ── in-distribution preprocessing (same crop the model was trained on) ────────
def preprocess_cropped(subject_dir: str, name: str, crop_target):
    """Load + normalise + brain-centre-crop one subject, returning the cropped
    4-channel tensor, the crop coordinates, and the original full volume shape.
    Mirrors BraTSDataset so inference stays in-distribution; the crop coordinates
    let us place the prediction back into native space."""
    raw = [nib.load(os.path.join(subject_dir, f"{name}-{m}.nii.gz")).get_fdata(dtype=np.float32)
           for m in MODALITIES]
    orig_shape = raw[0].shape
    union = np.zeros(orig_shape, dtype=bool)
    for a in raw:
        union |= a > 0
    coords = _compute_crop_coords(union, crop_target)
    cropped = [_pad_to_shape(coords.apply(_znorm(a)), crop_target) for a in raw]
    x = torch.from_numpy(np.stack(cropped, 0)[None]).float()        # (1,4,*crop_target)
    return x, coords, orig_shape


def uncrop(pred_cropped: np.ndarray, coords, orig_shape, crop_target) -> np.ndarray:
    """Embed a (C,*crop_target) prediction back into (C,*orig_shape) at `coords`,
    undoing any symmetric padding that was applied when the brain was smaller
    than the crop. Voxels outside the crop are background (no tumour)."""
    C = pred_cropped.shape[0]
    full = np.zeros((C,) + orig_shape, dtype=bool)
    win = (coords.x_end - coords.x_start, coords.y_end - coords.y_start, coords.z_end - coords.z_start)
    off = [(crop_target[i] - win[i]) // 2 for i in range(3)]   # padding removed before placing
    sub = pred_cropped[:, off[0]:off[0]+win[0], off[1]:off[1]+win[1], off[2]:off[2]+win[2]]
    full[:, coords.x_start:coords.x_end, coords.y_start:coords.y_end, coords.z_start:coords.z_end] = sub
    return full


# ── post-processing: remove small false-positive components ───────────────────
def remove_dust(pred_ov: np.ndarray, min_cc: int) -> np.ndarray:
    """Drop connected components smaller than `min_cc` voxels in each region (cc3d.dust).
    Removes the scattered false-positive specks that the model produces on the
    background-heavy full volume (BraTS post-processing, methodology section 10)."""
    if min_cc <= 0:
        return pred_ov
    out = np.empty_like(pred_ov)
    for c in range(pred_ov.shape[0]):
        out[c] = cc3d.dust(pred_ov[c].astype(np.uint8), threshold=min_cc,
                           connectivity=26).astype(bool)
    return out


# ── ground-truth region masks from a label volume ─────────────────────────────
def gt_regions(seg: np.ndarray):
    """Return (WT, TC, ET) binary masks from a BraTS label volume (1=NCR,2=ED,3=ET)."""
    wt = seg > 0
    tc = (seg == 1) | (seg == 3)
    et = seg == 3
    return [wt, tc, et]


# ── voxel metrics ─────────────────────────────────────────────────────────────
def voxel_dice(p, g):
    ps, gs = float(p.sum()), float(g.sum())
    if gs == 0 and ps == 0:
        return 1.0
    if gs == 0 or ps == 0:
        return 0.0
    return 2.0 * float((p & g).sum()) / (ps + gs)


def _hd95(p_bin, g_bin):
    """HD95 (mm) between two binary numpy masks via MONAI; assumes both non-empty."""
    p = torch.from_numpy(p_bin[None, None].astype(np.uint8)).float()
    g = torch.from_numpy(g_bin[None, None].astype(np.uint8)).float()
    return float(compute_hausdorff_distance(p, g, include_background=True, percentile=95).item())


def voxel_hd95(p, g):
    if p.sum() == 0 or g.sum() == 0:
        return None
    return _hd95(p, g)


# ── lesion-wise metric ────────────────────────────────────────────────────────
def _label_merged(mask: np.ndarray, dilation: int) -> np.ndarray:
    """Connected-component labels in which fragments of one lesion that lie within
    `dilation` voxels of each other are merged into a single component (following the
    BraTS 2023 lesion-wise definition), while the labels remain on the *original*
    voxels only. `dilation=0` falls back to plain 26-connectivity labelling."""
    if dilation <= 0:
        return cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(3, 1)
    dil = binary_dilation(mask, structure=struct, iterations=dilation)
    cc = cc3d.connected_components(dil.astype(np.uint8), connectivity=26)
    return cc * mask   # keep merged labels, restricted to the original lesion voxels


def lesionwise(pred_bin, gt_bin, hd95_penalty=374.0, min_lesion_vox=2, dilation=2):
    """Lesion-wise (DSC, HD95) for one region. See module docstring for the algorithm.
    `dilation` merges nearby fragments of a single lesion so that a tumour split into
    touching blobs is not penalised as multiple false-positive lesions."""
    gt_cc = _label_merged(gt_bin, dilation)
    pr_cc = _label_merged(pred_bin, dilation)

    def big_labels(cc):
        labs, counts = np.unique(cc, return_counts=True)
        return [int(l) for l, c in zip(labs, counts) if l != 0 and c >= min_lesion_vox]

    gt_labels = big_labels(gt_cc)
    pr_labels = set(big_labels(pr_cc))

    if not gt_labels and not pr_labels:
        return 1.0, 0.0  # both agree: no lesions

    dices, hds, matched_pred = [], [], set()
    for gl in gt_labels:
        gmask = gt_cc == gl
        overlap = set(int(x) for x in np.unique(pr_cc[gmask]) if x != 0) & pr_labels
        if overlap:
            matched_pred |= overlap
            pmask = np.isin(pr_cc, list(overlap))
            dices.append(voxel_dice(pmask, gmask))
            hds.append(_hd95(pmask, gmask))
        else:
            dices.append(0.0)       # false negative
            hds.append(hd95_penalty)
    for pl in pr_labels - matched_pred:   # false positives
        dices.append(0.0)
        hds.append(hd95_penalty)

    return float(np.mean(dices)), float(np.mean(hds))


def main():
    ap = argparse.ArgumentParser(description="BraTS 2023 full-volume + lesion-wise evaluation")
    ap.add_argument("--challenge", default="PED", choices=["GLI", "MEN", "PED"])
    ap.add_argument("--data_dir", required=True, help="Inner challenge training-data folder")
    ap.add_argument("--split_path", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--roi", type=int, default=128)
    ap.add_argument("--crop", type=int, nargs=3, default=[192, 192, 128],
                    help="Brain-centred crop the model was trained on (H W D)")
    ap.add_argument("--sw_batch_size", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--t_wt", type=float, default=0.45)
    ap.add_argument("--t_tc", type=float, default=0.40)
    ap.add_argument("--t_et", type=float, default=0.45)
    ap.add_argument("--hd95_penalty", type=float, default=374.0)
    ap.add_argument("--min_lesion_vox", type=int, default=2)
    ap.add_argument("--dilation", type=int, default=2,
                    help="Lesion-wise fragment-merge dilation in voxels (BraTS-style; 0 = strict)")
    ap.add_argument("--min_cc", type=int, default=50,
                    help="Post-processing: drop predicted components smaller than this many voxels (0 = off)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N subjects (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    roi = (args.roi, args.roi, args.roi)
    crop_target = tuple(args.crop)
    thr = (args.t_wt, args.t_tc, args.t_et)

    # subjects from the fold's validation split
    with open(args.split_path) as f:
        split = json.load(f)
    subjects = split["folds"][str(args.fold)]["val"]
    if args.limit > 0:
        subjects = subjects[:args.limit]

    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model = ckpt["model"]
    model.load_state_dict(ckpt["model_sd"])
    model = model.to(device).eval()
    print(f"Loaded epoch {ckpt.get('epoch')} | {args.challenge} fold {args.fold} | "
          f"evaluating {len(subjects)} subjects (full volume)\n")

    os.makedirs(args.out_dir, exist_ok=True)
    acc = {f"{m}_{r}": [] for m in ("vdsc", "vhd", "ldsc", "lhd") for r in REGION_NAMES}
    vhd_skip = {r: 0 for r in REGION_NAMES}
    failed = []

    # Open the CSV up front and flush each row, so a crash mid-run still leaves
    # the completed subjects on disk (the runs are memory-fragile on this host).
    fields = ["subject"] + [f"{m}_{r}" for r in REGION_NAMES for m in ("vdsc", "vhd", "ldsc", "lhd")]
    csv_path = os.path.join(args.out_dir, f"val_fullvol_fold{args.fold}_epoch{ckpt.get('epoch')}.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fields)
    writer.writeheader(); csv_f.flush()

    data_dir = args.data_dir
    with torch.no_grad():
        for i, name in enumerate(subjects):
            try:
                sd = os.path.join(data_dir, name)
                # In-distribution inference on the brain-centred crop, then uncrop.
                x, coords, orig_shape = preprocess_cropped(sd, name, crop_target)
                x = x.to(device)
                seg = nib.load(os.path.join(sd, f"{name}-seg.nii.gz")).get_fdata(dtype=np.float32)

                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = sliding_window_inference(x, roi, args.sw_batch_size, model,
                                                      overlap=args.overlap, mode="gaussian")
                probs = torch.sigmoid(logits.float())
                preds_disjoint = overlapping_probs_to_preds(probs, t1=thr[0], t2=thr[1], t3=thr[2]).float()
                pred_crop = disjoint_to_overlapping(preds_disjoint)[0].cpu().numpy().astype(bool)
                pred_ov = uncrop(pred_crop, coords, orig_shape, crop_target)
                pred_ov = remove_dust(pred_ov, args.min_cc)

                gts = gt_regions(seg.astype(np.int16))
                row = {"subject": name}
                for c, r in enumerate(REGION_NAMES):
                    p, g = pred_ov[c], gts[c]
                    vd = voxel_dice(p, g)
                    vh = voxel_hd95(p, g)
                    ld, lh = lesionwise(p, g, args.hd95_penalty, args.min_lesion_vox, args.dilation)
                    acc[f"vdsc_{r}"].append(vd); acc[f"ldsc_{r}"].append(ld); acc[f"lhd_{r}"].append(lh)
                    row[f"vdsc_{r}"] = round(vd, 4)
                    row[f"ldsc_{r}"] = round(ld, 4)
                    row[f"lhd_{r}"]  = round(lh, 2)
                    if vh is None:
                        vhd_skip[r] += 1; row[f"vhd_{r}"] = ""
                    else:
                        acc[f"vhd_{r}"].append(vh); row[f"vhd_{r}"] = round(vh, 2)
                writer.writerow(row); csv_f.flush()
                print(f"[{i+1:>3}/{len(subjects)}] {name} | "
                      + " ".join(f"{r} v{row[f'vdsc_{r}']:.2f}/L{row[f'ldsc_{r}']:.2f}" for r in REGION_NAMES),
                      flush=True)
                del x, logits, probs, preds_disjoint, pred_crop, pred_ov, gts, seg
            except Exception as exc:
                failed.append(name)
                print(f"[{i+1:>3}/{len(subjects)}] {name} | SKIPPED: {type(exc).__name__}: {exc}", flush=True)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    csv_f.close()

    # summary
    print("\n" + "=" * 68)
    print(f"FULL-VOLUME EVALUATION — {args.challenge} fold {args.fold} (epoch {ckpt.get('epoch')})")
    print("=" * 68)
    print(f"{'Region':<7}{'voxel DSC':<12}{'voxel HD95':<13}{'lesion DSC':<13}{'lesion HD95':<12}")
    n_done = len(acc["vdsc_WT"])
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    for r in REGION_NAMES:
        print(f"{r:<7}{mean(acc[f'vdsc_{r}']):<12.4f}{mean(acc[f'vhd_{r}']):<13.2f}"
              f"{mean(acc[f'ldsc_{r}']):<13.4f}{mean(acc[f'lhd_{r}']):<12.2f}")
    print("-" * 68)
    print(f"Subjects scored: {n_done}/{len(subjects)} | skipped: {len(failed)}")
    print(f"Mean voxel DSC : {mean([mean(acc[f'vdsc_{r}']) for r in REGION_NAMES]):.4f}")
    print(f"Mean lesion DSC: {mean([mean(acc[f'ldsc_{r}']) for r in REGION_NAMES]):.4f}")
    print(f"voxel-HD95 skipped (empty): " + ", ".join(f"{r}={vhd_skip[r]}" for r in REGION_NAMES))
    if failed:
        print(f"SKIPPED subjects: {', '.join(failed)}")
    print(f"\nPer-subject metrics: {csv_path}")


if __name__ == "__main__":
    main()
