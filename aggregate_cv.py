"""
Aggregate per-fold full-volume metrics into a single 5-fold cross-validation result.

Reads output/<CH>_swin_fold{0..4}/val_fullvol_fold{f}_epoch*.csv (written by
evaluate_fullvol.py) and reports, per region, the pooled mean +/- std over all
validation subjects across the folds (each subject is held out exactly once, so the
pool covers the whole dataset), plus the per-fold means. This is the robust 5-fold
number that replaces the single-fold estimate.

  python aggregate_cv.py --challenge GLI
"""
import csv
import glob
import argparse
import statistics as st

REGIONS = ["WT", "TC", "ET"]
METRICS = [("vdsc", "voxel DSC"), ("ldsc", "lesion DSC"), ("vhd", "voxel HD95"), ("lhd", "lesion HD95")]


def load(challenge):
    per_fold, pooled = {}, {m: {r: [] for r in REGIONS} for m, _ in METRICS}
    paths = sorted(glob.glob(f"output/{challenge}_swin_fold*/val_fullvol_fold*_epoch*.csv"))
    if not paths:
        raise SystemExit(f"No val_fullvol CSVs found for {challenge}. Run evaluate_fullvol per fold first.")
    for p in paths:
        fold = p.split("_swin_fold")[1].split("/")[0].split("\\")[0]
        rows = list(csv.DictReader(open(p)))
        fm = {}
        for m, _ in METRICS:
            for r in REGIONS:
                vals = [float(x[f"{m}_{r}"]) for x in rows if x.get(f"{m}_{r}") not in (None, "", "nan")]
                pooled[m][r] += vals
                fm[f"{m}_{r}"] = st.mean(vals) if vals else float("nan")
        per_fold[fold] = (len(rows), fm)
    return paths, per_fold, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge", required=True, choices=["GLI", "MEN", "PED"])
    args = ap.parse_args()

    paths, per_fold, pooled = load(args.challenge)
    total = sum(n for n, _ in per_fold.values())

    print(f"\n=== {args.challenge}: 5-fold cross-validation ({len(per_fold)} folds, {total} subjects) ===")
    print(f"{'metric':<12}{'WT':>16}{'TC':>16}{'ET':>16}{'mean':>10}")
    for m, label in METRICS:
        cells, means = [], []
        for r in REGIONS:
            vals = pooled[m][r]
            if vals:
                mu, sd = st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)
                means.append(mu)
                cells.append(f"{mu:.3f}±{sd:.3f}")
            else:
                cells.append("—")
        mean = f"{st.mean(means):.3f}" if means else "—"
        print(f"{label:<12}" + "".join(f"{c:>16}" for c in cells) + f"{mean:>10}")

    print("\nPer-fold mean DSC (voxel / lesion):")
    for fold in sorted(per_fold):
        n, fm = per_fold[fold]
        v = st.mean([fm[f"vdsc_{r}"] for r in REGIONS])
        l = st.mean([fm[f"ldsc_{r}"] for r in REGIONS])
        print(f"  fold {fold} (n={n}):  voxel {v:.3f}  |  lesion {l:.3f}")
    print(f"\nSources: {len(paths)} CSV file(s).")


if __name__ == "__main__":
    main()
