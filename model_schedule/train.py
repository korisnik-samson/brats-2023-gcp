import os
import numpy as np
import torch
from torch import optim
import csv

from utils.model_utils import (
    load_or_initialize_training, make_dataloader, warmup_cosine_lr, train_one_epoch,
)

def train(data_dir, model, loss_functions, loss_weights, init_lr, max_epoch, training_regions='overlapping',
          out_dir=None, backup_interval=10, batch_size=1, device='cuda',
          challenge='GLI', fold=0, split_path=None, transform=None, cache_dir=None,
          crop_target=(192, 192, 128), num_workers=0, weight_decay=1e-5, warmup_epochs=5,
          use_amp=True, max_grad_norm=1.0, init_from=None, freeze_encoder=False):
    """Runs the training routine (AdamW + warmup/cosine LR + mixed precision).

    Args:
        data_dir: Directory of training data (inner subject folder).
        model: The PyTorch model to be trained.
        loss_functions: List of loss functions to be used for training.
        loss_weights: List of weights corresponding to each loss function.
        init_lr: Peak learning rate (reached at the end of warmup).
        max_epoch: Maximum number of epochs to train for.
        training_regions: Whether training on 'disjoint' or 'overlapping' regions.
        out_dir: Directory to save model checkpoints and loss values.
        backup_interval: How often (in epochs) to save a backup checkpoint.
        batch_size: Batch size of dataloader.
        challenge: 'GLI', 'MEN', or 'PED'.
        fold: CV fold index used as the validation split.
        split_path: Path to the split JSON from SplitManager.generate().
        transform: MONAI transform (patch crop + augmentation) applied per sample.
        cache_dir: Directory for cached preprocessed full volumes.
        crop_target: Full-volume size cached before patch cropping.
        num_workers: DataLoader worker processes.
        weight_decay: AdamW weight decay.
        warmup_epochs: Linear LR warmup length before cosine decay.
        use_amp: Enable CUDA mixed-precision training.
        max_grad_norm: Gradient-norm clip threshold (None to disable).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    loss_functions = [lf.to(device) for lf in loss_functions]

    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()

    os.makedirs(out_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    training_loss_path = os.path.join(out_dir, 'training_loss.csv')
    backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
    os.makedirs(backup_ckpts_dir, exist_ok=True)

    use_amp = bool(use_amp) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print("---------------------------------------------------")
    print("TRAINING SUMMARY")
    print(f"Data directory: {data_dir}")
    print(f"Challenge: {challenge} | Fold: {fold}")
    print(f"Model: {model}")
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Loss functions: {loss_functions}")
    print(f"Loss weights: {loss_weights}")
    print(f"Peak learning rate: {init_lr} | weight_decay: {weight_decay}")
    print(f"Max epochs: {max_epoch} | warmup: {warmup_epochs}")
    print(f"Training regions: {training_regions}")
    print(f"Out directory: {out_dir}")
    print(f"Backup interval: {backup_interval}")
    print(f"Batch size: {batch_size} | num_workers: {num_workers}")
    print(f"Split path: {split_path} | cache_dir: {cache_dir}")
    print("---------------------------------------------------")

    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)

    # Check if training for first time or continuing from a saved checkpoint.
    epoch_start = load_or_initialize_training(model, optimizer, latest_ckpt_path)

    # Transfer learning: warm-start the weights from another checkpoint (e.g. a
    # GLI-trained model) when starting fresh. Optimiser/epoch are NOT inherited —
    # fine-tuning uses a new schedule. Loaded on CPU then freed to spare VRAM.
    if init_from and epoch_start == 1:
        print(f'Warm-starting model weights from: {init_from}')
        init_ckpt = torch.load(init_from, map_location='cpu', weights_only=False)
        model.load_state_dict(init_ckpt['model_sd'])
        del init_ckpt
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if freeze_encoder:
            frozen = 0
            for name, p in model.named_parameters():
                if 'swinViT' in name or 'encoder' in name:
                    p.requires_grad = False
                    frozen += 1
            print(f'Froze {frozen} encoder parameter tensors (decoder fine-tuned only).')

    # Restore AMP scaler state on resume so loss scaling continues smoothly.
    # Load to CPU and free immediately — otherwise the full 750MB checkpoint
    # (model + optimizer tensors) would sit duplicated on the GPU for the whole
    # run and cause an out-of-memory error during backward.
    if scaler is not None and os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location='cpu', weights_only=False)
        if ckpt.get('scaler_sd') is not None:
            scaler.load_state_dict(ckpt['scaler_sd'])
        del ckpt
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    train_loader = make_dataloader(
        data_dir, shuffle=True, mode='train', batch_size=batch_size,
        challenge=challenge, fold=fold, split_path=split_path, transform=transform,
        cache_dir=cache_dir, crop_target=crop_target, num_workers=num_workers,
    )

    print(f'Training starts on {len(train_loader.dataset)} subjects.\n')

    for epoch in range(epoch_start, max_epoch + 1):
        print(f'Starting epoch {epoch}...')

        lr = warmup_cosine_lr(optimizer, epoch, max_epoch, init_lr, warmup=warmup_epochs)

        average_epoch_loss = train_one_epoch(
            model, optimizer, train_loader,
            loss_functions, loss_weights, training_regions, device,
            scaler=scaler, max_grad_norm=max_grad_norm,
            epoch=epoch, total_epochs=max_epoch,
        )

        # Save and report loss from the epoch.
        save_tloss_csv(training_loss_path, epoch, average_epoch_loss, lr)

        print(f'Epoch {epoch} completed. Average loss = {average_epoch_loss:.4f} | lr = {lr:.2e}.')
        print('Saving model checkpoint...')

        checkpoint = {
            'epoch': epoch,
            'model_sd': model.state_dict(),
            'optim_sd': optimizer.state_dict(),
            'scaler_sd': scaler.state_dict() if scaler is not None else None,
            'model': model,
            'loss_functions': loss_functions,
            'loss_weights': loss_weights,
            'init_lr': init_lr,
            'training_regions': training_regions,
            'challenge': challenge,
            'fold': fold,
        }

        torch.save(checkpoint, latest_ckpt_path)

        if epoch % backup_interval == 0:
            torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar'))

        print('Checkpoint saved successfully.')

def save_tloss_csv(pathname, epoch, tloss, lr=None):
    write_header = not os.path.exists(pathname)
    with open(pathname, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow(['Epoch', 'Training Loss', 'Learning Rate'])

        writer.writerow([epoch, tloss, lr])

if __name__ == '__main__':

    from models import unet_3d
    import torch.nn as nn

    data_dir = '../data'
    model = unet_3d.UNet3D()
    loss_functions = [nn.MSELoss(), nn.CrossEntropyLoss()]

    loss_weights = [0.4, 0.7]
    lr = 6e-5
    max_epoch = 20

    out_dir = '../data/output'

    train(data_dir, model, loss_functions, loss_weights, lr, max_epoch, out_dir=out_dir)