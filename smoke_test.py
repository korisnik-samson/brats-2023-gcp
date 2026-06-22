"""One-batch smoke test: verifies the PED pipeline + 12GB fit before long runs."""
import time
import torch
from torch.utils.data import DataLoader

from loader.brats_dataset import BraTSDataset
from models.swin_unetr_3d import SwinUNETR3D
from losses.loss_functions import DiceLoss, FocalLoss
from utils.model_utils import compute_loss
from utils.general_utils import seg_to_one_hot_channels, disjoint_to_overlapping
from train_swin_unetr import build_train_transform

DATA = "dataset/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"
device = torch.device("cuda")

tf = build_train_transform((128, 128, 128), fg_prob=0.9)
ds = BraTSDataset(DATA, challenge="PED", mode="train", fold=0,
                  split_path="splits/PED_5fold_split.json", transform=tf,
                  cache_dir="cache/PED", crop_target=(192, 192, 128))
print("dataset size:", len(ds))

loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

t0 = time.time()
name, imgs, seg = next(iter(loader))
print(f"batch loaded in {time.time()-t0:.1f}s | names={name}")
print("n modalities:", len(imgs), "| img shape:", tuple(imgs[0].shape), "| seg shape:", tuple(seg.shape))

model = SwinUNETR3D(img_size=(128, 128, 128), in_channels=4, out_channels=3,
                    feature_size=48, use_checkpoint=True).to(device)
print(model)

loss_fns = [DiceLoss().to(device), FocalLoss().to(device)]
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")

imgs = [im.to(device) for im in imgs]
seg = seg.to(device)
seg = disjoint_to_overlapping(seg_to_one_hot_channels(seg))
x_in = torch.cat(imgs, dim=1)
print("x_in:", tuple(x_in.shape), "| seg(overlap):", tuple(seg.shape))

model.train()
opt.zero_grad(set_to_none=True)
t0 = time.time()
with torch.autocast("cuda", dtype=torch.float16):
    out = model(x_in)
out = out.float()
loss = compute_loss(out, seg, loss_fns, [1.0, 1.0], device)
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt)
scaler.update()
torch.cuda.synchronize()
print(f"one train step in {time.time()-t0:.2f}s | loss={loss.item():.4f}")
print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("SMOKE TEST PASSED")
