"""This module contains utility functions for training models."""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:                       # graceful fallback if tqdm is absent
    def tqdm(iterable, *args, **kwargs):
        return iterable

from loader import brats_dataset
from utils.general_utils import seg_to_one_hot_channels, disjoint_to_overlapping

def load_or_initialize_training(model, optimizer, latest_ckpt_path, train_with_val=False):
    """Loads training checkpoint if it exists, or initializes training from scratch.

    Args:
        model: The PyTorch model to be trained.
        optimizer: The optimizer used for training.
        latest_ckpt_path: The path to the latest model checkpoint.
        train_with_val: If True, also returns best saved validation loss and dice. Defaults to False.

    Returns:
        The starting epoch number.
        If 'train_with_val' is True, also returns best saved validation loss and dice.
    """

    if not os.path.exists(latest_ckpt_path):
        epoch_start = 1

        if train_with_val:
            best_vloss = float('inf')
            best_dice = 0
        print('No training checkpoint found. Will start training from scratch.')

    else:
        print('Training checkpoint found. Loading checkpoint...')

        # Load to CPU first. The checkpoint was saved while the model was on the
        # GPU, so without map_location all its tensors (model_sd, optim_sd, and a
        # duplicate full model object) would land on the GPU, fragmenting VRAM and
        # causing an out-of-memory error on resume. weights_only=False is required
        # because our checkpoints store full (trusted) model/loss objects.
        checkpoint = torch.load(latest_ckpt_path, map_location='cpu', weights_only=False)
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_sd'])
        optimizer.load_state_dict(checkpoint['optim_sd'])

        # Move optimizer state onto the model's device (load_state_dict above put
        # it on CPU; AdamW needs its moments on the same device as the params).
        param_device = next(model.parameters()).device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(param_device)

        if train_with_val:
            best_vloss = checkpoint['vloss']
            best_dice = checkpoint['dice']

        # Free the CPU-side checkpoint copy before training starts.
        del checkpoint

        print(f'Checkpoint loaded. Will continue training from epoch {epoch_start}.')

    if train_with_val:
        return epoch_start, best_vloss, best_dice

    return epoch_start

def make_dataloader(data_dir, shuffle, mode, batch_size=1, do_crop=True,
                    challenge='GLI', fold=0, split_path=None, transform=None,
                    cache_dir=None, crop_target=(192, 192, 128), num_workers=0):
    """Creates dataloader for provided data directory.

    Args:
        data_dir: Directory of challenge training data (inner subject folder).
        shuffle: Whether to shuffle the dataset each epoch.
        mode: 'train', 'val', or 'test'.
        batch_size: Batch size. Defaults to 1.
        do_crop: Whether the dataset applies foreground cropping. Defaults to True.
        challenge: 'GLI', 'MEN', or 'PED'. Controls the subject-directory prefix.
        fold: CV fold index used as the validation split.
        split_path: Path to the split JSON from SplitManager.generate().
        transform: MONAI-style transform applied to {'image','label'} (patch
            cropping + augmentation). Run fresh every epoch.
        cache_dir: If set, preprocessed full volumes are cached as .pt files
            (~20x faster per epoch after the first pass).
        crop_target: Full-volume spatial size cached/returned before patching.
        num_workers: DataLoader worker processes. 0 is safest on Windows for the
            first (cache-building) pass.
    """
    dataset = brats_dataset.BraTSDataset(
        data_dir,
        challenge=challenge,
        mode=mode,
        fold=fold,
        split_path=split_path,
        transform=transform,
        cache_dir=cache_dir,
        crop_target=crop_target,
        do_crop=do_crop,
    )
    # pin_memory=False: page-locked host memory counts against the Windows
    # commit limit and cannot be paged out — avoid it on this RAM-constrained box.
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    return dataloader

def exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate):
    """Exponentially decays learning rate of optimizer at given epoch."""
    lr = init_lr * (decay_rate ** (epoch - 1))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_cosine_lr(optimizer, epoch, max_epoch, init_lr, warmup=5, min_lr_frac=0.01):
    """Linear warmup for `warmup` epochs, then cosine decay to `min_lr_frac * init_lr`.

    Computed purely from the epoch number so it is resume-safe (no scheduler
    state to persist across checkpoints).
    """
    import math
    if epoch <= warmup:
        lr = init_lr * epoch / max(1, warmup)
    else:
        progress = (epoch - warmup) / max(1, (max_epoch - warmup))
        progress = min(max(progress, 0.0), 1.0)
        lr = init_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_loss(output, seg, loss_functs, loss_weights, device):
    """Computes weighted loss between model output and ground truth, summed across each region."""
    loss = 0.

    for n, loss_function in enumerate(loss_functs):
        temp = 0

        for i in range(3):
            temp += loss_function(output[:, i:i + 1].to(device), seg[:, i : i+1].to(device))

        loss += temp * loss_weights[n]

    return loss

def train_one_epoch(model, optimizer, train_loader, loss_functions, loss_weights, training_regions, device,
                    scaler=None, max_grad_norm=1.0, epoch=None, total_epochs=None):
    """Performs one training loop of model according to given optimizer, loss functions and associated weights.
    Args:
        model: The PyTorch model to be trained.
        optimizer: The optimizer used for training.
        train_loader: The dataloader for training data.
        loss_functions: List of loss functions.
        loss_weights: List of associated weightings for each loss function.
        training_regions: String specifying whether 'disjoint' or 'overlapping' regions will be used for training.
        scaler: Optional torch.amp.GradScaler. If provided, the forward pass runs
            under autocast (mixed precision); the loss is computed in fp32 to
            avoid fp16 overflow in the Dice voxel-count reductions.
        max_grad_norm: Gradient-norm clipping threshold. Set to None to disable.
    Returns:
        The average training loss over the epoch.
    """
    losses_over_epoch = []

    model.train()

    desc = f"Epoch {epoch}/{total_epochs}" if epoch is not None else "Training"
    pbar = tqdm(train_loader, desc=desc, total=len(train_loader),
                file=sys.stdout, dynamic_ncols=True, leave=False)

    for _, imgs, seg in pbar:

        # Move data to GPU.
        imgs = [img.to(device, non_blocking=True) for img in imgs]  # each B1HWD
        seg = seg.to(device, non_blocking=True)

        # Split segmentation into 3 disjoint one-hot channels (B3HWD).
        seg = seg_to_one_hot_channels(seg)

        if training_regions == 'overlapping':
            # Convert to overlapping WT / TC / ET encoding (B3HWD).
            seg = disjoint_to_overlapping(seg)

        x_in = torch.cat(imgs, dim=1)  # x_in is B4HWD

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed precision: autocast the forward, keep the loss in fp32.
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(x_in)
            output = output.float()
            loss = compute_loss(output, seg, loss_functions, loss_weights, device)

            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x_in).float()
            loss = compute_loss(output, seg, loss_functions, loss_weights, device)

            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        losses_over_epoch.append(loss.detach().cpu())
        pbar.set_postfix(loss=f"{np.mean(losses_over_epoch):.4f}")

    # Compute loss from the epoch.
    average_epoch_loss = np.mean(losses_over_epoch)

    return average_epoch_loss


def freeze_layers(model, frozen_layers):
    """Freezes specified model layers. Afterwards parameters in these layers will not be updated when training.

    Args:
        model: The model to be trained.
        frozen_layers: List of strings specifying model layers.
    """

    for name, param in model.named_parameters():
        needs_freezing = False

        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break

        if needs_freezing:
            print(f'Freezing parameter {name}.')
            param.requires_grad = False

def check_frozen(model, frozen_layers):
    """Iterates through model layers and checks whether specified layers are frozen.

    Args:
        model: The model to be trained.
        frozen_layers: List of strings specifying model layers.
    """
    for name, param in model.named_parameters():
        needs_freezing = False

        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break

        if needs_freezing:
            if param.requires_grad:
                print(f'Warning! Param {name} should not require grad but does.')
                break
            else:
                print(f'Parameter {name} is frozen.')

# Example parts of unet_3d model to freeze
# 'encoder': ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Conv7'],
# 'decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13', 'Up2', 'Up_conv2', 'Conv_1x12', 'Up1', 'Up_conv1', 'Conv_1x11'],
# 'middle' : ['Conv5', 'Conv6', 'Conv7', 'Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4'],
# 'none' : [],
# 'deep_decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4']