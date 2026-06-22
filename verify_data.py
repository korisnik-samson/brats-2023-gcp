"""
Pre-flight data integrity check — run after download_data.sh, before the 5-fold runs.

For each challenge it confirms the training directory exists at the path the launchers
expect, that every subject folder has all four modalities + the segmentation, and that
every subject named in the committed split JSON is present and complete on disk. Optionally
checks the official ValidationData (modalities only, no seg). Stdlib only — no torch/monai
needed, so it runs even before the full environment is installed.

  python verify_data.py            # all challenges, training data
  python verify_data.py --val      # also check the ValidationData folders
  python verify_data.py --challenge PED
"""
import os
import sys
import json
import glob
import argparse

MODALITIES = ("t1c", "t1n", "t2f", "t2w")

# canonical paths created by download_data.sh / matching the launchers + splits
PATHS = {
    "GLI": dict(
        train="dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        val="dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData",
        split="splits/GLI_5fold_split.json"),
    "MEN": dict(
        train="dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train",
        val="dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData",
        split="splits/MEN_5fold_split.json"),
    "PED": dict(
        train="dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData",
        val="dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData",
        split="splits/PED_5fold_split.json"),
}


def files_ok(subject_dir, name, need_seg=True):
    """Return list of missing/empty required files for one subject."""
    missing = []
    needed = list(MODALITIES) + (["seg"] if need_seg else [])
    for suf in needed:
        f = os.path.join(subject_dir, f"{name}-{suf}.nii.gz")
        if not (os.path.isfile(f) and os.path.getsize(f) > 0):
            missing.append(f"{suf}")
    return missing


def check_dir(challenge, data_dir, need_seg):
    """Scan a directory of subject folders; return (n_total, incomplete:list[(name,missing)])."""
    subs = sorted(d for d in glob.glob(os.path.join(data_dir, f"BraTS-{challenge}-*")) if os.path.isdir(d))
    incomplete = []
    for d in subs:
        name = os.path.basename(d)
        m = files_ok(d, name, need_seg=need_seg)
        if m:
            incomplete.append((name, m))
    return len(subs), incomplete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge", choices=list(PATHS), help="Only this challenge (default: all)")
    ap.add_argument("--val", action="store_true", help="Also check the ValidationData folders")
    args = ap.parse_args()

    challenges = [args.challenge] if args.challenge else list(PATHS)
    all_ok = True

    for ch in challenges:
        p = PATHS[ch]
        print(f"\n=== {ch} ===")
        # ---- training data ----
        if not os.path.isdir(p["train"]):
            print(f"  ✗ training dir MISSING: {p['train']}")
            all_ok = False
            continue
        n, incomplete = check_dir(ch, p["train"], need_seg=True)
        print(f"  training subjects on disk : {n}")
        if incomplete:
            all_ok = False
            print(f"  ✗ {len(incomplete)} incomplete subject(s) (missing files):")
            for name, miss in incomplete[:8]:
                print(f"      {name}: missing {', '.join(miss)}")
            if len(incomplete) > 8:
                print(f"      ... and {len(incomplete) - 8} more")
        else:
            print(f"  ✓ all {n} subjects have 4 modalities + seg")

        # ---- cross-check against the committed split ----
        if os.path.isfile(p["split"]):
            split_subjects = set(json.load(open(p["split"])).get("subjects", {}))
            on_disk = {os.path.basename(d) for d in glob.glob(os.path.join(p["train"], f"BraTS-{ch}-*"))}
            missing = sorted(split_subjects - on_disk)
            print(f"  split lists {len(split_subjects)} subjects; missing on disk: {len(missing)}")
            if missing:
                all_ok = False
                print(f"      e.g. {', '.join(missing[:5])}")
        else:
            print(f"  ! split file not found: {p['split']}")

        # ---- validation data (optional, no seg) ----
        if args.val:
            if os.path.isdir(p["val"]):
                nv, inc_v = check_dir(ch, p["val"], need_seg=False)
                msg = "✓" if not inc_v else "✗"
                print(f"  {msg} validation subjects: {nv} ({len(inc_v)} missing modalities)")
                if inc_v:
                    all_ok = False
            else:
                print(f"  ! validation dir not found: {p['val']}")

    print("\n" + ("=" * 40))
    print("RESULT:", "✓ ALL CHECKS PASSED" if all_ok else "✗ PROBLEMS FOUND — see above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
