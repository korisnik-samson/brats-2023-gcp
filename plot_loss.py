"""
Plot the training-loss curve from a run's training_loss.csv.
Lightweight: CSV + matplotlib only (no torch / no GPU), safe to run while training.

Usage:
    python plot_loss.py                       # defaults to output/PED_swin_fold0
    python plot_loss.py --run output/GLI_swin_fold0
"""
import os
import csv
import argparse

import matplotlib
matplotlib.use("Agg")  # headless: write a PNG, no GUI window
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="output/PED_swin_fold0", help="Run output directory")
    args = ap.parse_args()

    csv_path = os.path.join(args.run, "training_loss.csv")

    if not os.path.exists(csv_path):
        raise SystemExit(f"No training_loss.csv found in {args.run}")

    epochs, losses, lrs = [], [], []

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                epochs.append(int(row["Epoch"]))
                losses.append(float(row["Training Loss"]))
                lrs.append(float(row["Learning Rate"]) if row.get("Learning Rate") else None)
            except (ValueError, KeyError):
                continue

    if not epochs:
        raise SystemExit("CSV has no completed epochs yet.")

    best_i = min(range(len(losses)), key=lambda i: losses[i])

    print(f"Run: {args.run}")
    print(f"Epochs completed : {epochs[-1]}")
    print(f"Latest loss      : {losses[-1]:.4f}  (epoch {epochs[-1]})")
    print(f"Best  loss       : {losses[best_i]:.4f}  (epoch {epochs[best_i]})")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, losses, color="tab:blue", lw=1.8, label="Training loss")
    ax1.scatter([epochs[best_i]], [losses[best_i]], color="tab:red", zorder=5, label=f"best {losses[best_i]:.3f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="tab:blue")
    ax1.grid(alpha=0.3)

    if any(v is not None for v in lrs):
        ax2 = ax1.twinx()
        ax2.plot(epochs, [v if v is not None else float("nan") for v in lrs], color="tab:green", lw=1.0, alpha=0.6, label="LR")
        ax2.set_ylabel("Learning rate", color="tab:green")

    ax1.set_title(f"{os.path.basename(args.run)} — training loss ({epochs[-1]} epochs)")
    ax1.legend(loc="upper right")
    fig.tight_layout()

    out_png = os.path.join(args.run, "loss_curve.png")
    fig.savefig(out_png, dpi=120)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
