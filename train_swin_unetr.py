"""
Training Entry Point for BraTS 2023 with Swin UNETR
===================================================
Initializes the Swin UNETR 3D model, sets up the BraTS 2023 dataset (GLI, MEN,
or PED) with patch-based sampling + on-the-fly augmentation, and starts the
training loop with a hybrid Dice + Focal loss under mixed precision.

Designed to fit a single 12 GB GPU (RTX 4070):
  * 128^3 foreground-biased patches (RandCropByPosNegLabeld)
  * gradient checkpointing in Swin UNETR (use_checkpoint=True)
  * AMP (autocast + GradScaler), AdamW, warmup + cosine LR
  * disk caching of preprocessed full volumes (~20x faster per epoch)

Usage:
    python train_swin_unetr.py --challenge PED \
        --data_dir dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData \
        --split_path splits/PED_5fold_split.json --fold 0 \
        --cache_dir cache/PED --out_dir output/PED_swin_fold0 --epochs 300
"""

import os
import argparse
import random

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from models.swin_unetr_3d import SwinUNETR3D
from models.unet_3d import UNet3D
from model_schedule.train import train
from losses.loss_functions import DiceLoss, FocalLoss


def set_seed(seed: int = 42):
    """Deterministic seeding for reproducibility (methodology section 13.1)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(patch_size, fg_prob=0.9):
    """Patch-based training transform.

    RandCropByPosNegLabeld returns a list (one dict per sample); we request a
    single sample and unwrap it so BraTSDataset's single-dict contract holds.
    """
    pos = fg_prob
    neg = 1.0 - fg_prob

    compose = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=pos,
            neg=neg,
            num_samples=1,
            image_key="image",
            allow_smaller=True,
        ),

        # Spatial augmentation (applied identically to image + label)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),

        # Intensity augmentation (image channels only)
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.05),
    ])

    def transform(sample):
        out = compose(sample)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    return transform


def main():
    parser = argparse.ArgumentParser(description="BraTS 2023 Swin UNETR Training")
    parser.add_argument("--challenge", type=str, default="PED", choices=["GLI", "MEN", "PED"], help="Sub-challenge to train on")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold index (0-4)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the challenge training data directory (inner subject folder)")
    parser.add_argument("--split_path", type=str, default=None, help="Path to the split JSON file")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save checkpoints and logs")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for cached preprocessed volumes")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training (1 fits 128^3 patches in 12GB)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate (after warmup)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs")
    parser.add_argument("--backup_interval", type=int, default=10, help="Save a numbered backup checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 safest on Windows)")
    parser.add_argument("--model", type=str, default="swin", choices=["swin", "unet"],
                        help="Architecture: 'swin' (Swin UNETR) or 'unet' (3D U-Net baseline, H2)")
    parser.add_argument("--feature_size", type=int, default=48, help="Swin UNETR embedding feature size")
    parser.add_argument("--patch", type=int, default=128, help="Cubic training patch size")
    parser.add_argument("--fg_prob", type=float, default=0.9, help="Probability a sampled patch is centred on tumour")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable Swin gradient checkpointing (faster; needs more VRAM — fine on a 24GB L4)")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Warm-start model weights from this checkpoint (transfer learning)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze the Swin encoder when warm-starting (fine-tune decoder only)")

    args = parser.parse_args()
    set_seed(args.seed)

    patch_size = (args.patch, args.patch, args.patch)

    # 1. Transforms (patch sampling + augmentation)
    train_transform = build_train_transform(patch_size, fg_prob=args.fg_prob)

    # 2. Model
    if args.model == "swin":
        # Gradient checkpointing is mandatory for 12 GB VRAM.
        model = SwinUNETR3D(
            img_size=patch_size,
            in_channels=4,
            out_channels=3,
            feature_size=args.feature_size,
            use_checkpoint=not args.no_checkpoint,
        )
    else:  # 'unet' — 3D U-Net baseline for the architecture comparison (H2)
        model = UNet3D(img_ch=4, output_ch=3)

    # 3. Hybrid loss: Dice (overlap) + Focal (class imbalance)
    loss_functions = [DiceLoss(), FocalLoss()]
    loss_weights = [1.0, 1.0]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Starting Swin UNETR training for {args.challenge} (fold {args.fold})...")

    train(
        data_dir=args.data_dir,
        model=model,
        loss_functions=loss_functions,
        loss_weights=loss_weights,
        init_lr=args.lr,
        max_epoch=args.epochs,
        training_regions='overlapping',
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        backup_interval=args.backup_interval,
        challenge=args.challenge,
        fold=args.fold,
        split_path=args.split_path,
        transform=train_transform,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        use_amp=not args.no_amp,
        init_from=args.init_from,
        freeze_encoder=args.freeze_encoder,
    )


if __name__ == "__main__":
    main()
