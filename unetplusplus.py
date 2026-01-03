import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    AsDiscrete,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# ---------------------------------------------------------
# 1. UNet++ Architecture (3D Version)
# ---------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, deep_supervision=False):
        super(UNetPlusPlus3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        
        # Filters typically [32, 64, 128, 256, 512] for 3D to save VRAM, 
        # or [64, 128, 256, 512, 1024] if you have high-end GPUs.
        # Using your original scale [64, 128...] might be heavy for UNet++, 
        # so I am using standard filters.
        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool3d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # ----------------- Encoder (Backbone) -----------------
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # ----------------- Nested Layers -----------------
        
        # Level 0, Column 1
        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        
        # Level 1, Column 1
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        # Level 0, Column 2
        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        
        # Level 2, Column 1
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        # Level 1, Column 2
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        # Level 0, Column 3
        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        
        # Level 3, Column 1
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        # Level 2, Column 2
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])
        # Level 1, Column 3
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])
        # Level 0, Column 4 (Output Node for Standard UNet++)
        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        # ----------------- Final Segmentation Layers -----------------
        self.final = nn.Conv3d(nb_filter[0], out_channels, kernel_size=1)
        
        if self.deep_supervision:
            self.final1 = nn.Conv3d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, input):
        # --- Backbone ---
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # --- Nest Level 1 ---
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # --- Nest Level 2 ---
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # --- Nest Level 3 ---
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # --- Nest Level 4 ---
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final(x0_4)
            # In deep supervision, you typically compute loss for all and sum them
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)

# ---------------------------------------------------------
# 2. Setup & Configuration
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model
model = UNetPlusPlus3D(in_channels=1, out_channels=3, deep_supervision=False).to(device)

# Initialize Optimizer
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# ---------------------------------------------------------
# 3. Data Loading Pipeline
# ---------------------------------------------------------

proc_dir = "/home/Task07_Pancreas"
# Ensure these paths exist on your system
train_images = sorted(glob.glob(os.path.join(proc_dir, "imagesTr", "pancreas*.nii")))
train_labels = sorted(glob.glob(os.path.join(proc_dir, "labelsTr", "pancreas*.nii")))

print(f"total images: {len(train_images)}, total labels: {len(train_labels)}")

data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]

val_split_idx = int(len(data_dicts) * 0.9)
train_files, val_files = data_dicts[:val_split_idx], data_dicts[val_split_idx:]
print(f"train size: {len(train_files)}, validation size: {len(val_files)}")

# Transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1, neg=1,
        num_samples=2,
        image_key="image",
        image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.1, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.1, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# ---------------------------------------------------------
# 4. Training Loop setup
# ---------------------------------------------------------

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)

max_epochs = 300
accumulation_steps = 5
val_interval = 5
best_metric = -1
best_metric_epoch = -1

# ---> Initialize Scheduler <---
# T_max is usually set to the maximum number of epochs.
scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

log_filename = "/workspace/unetplusplus_training_log.csv"
with open(log_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_dice", "best_dice", "lr"])

print(f"Logging metrics to {log_filename}")

scaler = torch.amp.GradScaler(device="cuda")

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------

for epoch in range(max_epochs):
    print(f"-" * 10)
    print(f"Epoch {epoch + 1}/{max_epochs}")
    
    model.train()
    epoch_loss = 0
    step = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    optimizer.zero_grad() 

    for i, batch_data in enumerate(progress_bar):
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss = loss / accumulation_steps 
        
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 

        current_loss = loss.item() * accumulation_steps
        epoch_loss += current_loss
        
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

    # Handle any remaining gradients
    if (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss /= step
    print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # ---> Step the scheduler <---
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    print(f"Current Learning Rate: {current_lr:.6f}")

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------
    current_val_metric = ""
    if (epoch + 1) % val_interval == 0:
        print(f"-----------------Validation-----------------")
        model.eval()
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validation", leave=False)
            
            for val_data in val_progress:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                inference_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=inference_model,
                        sw_device=device,
                        device=device,
                    )
                
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list = decollate_batch(val_labels)
                
                val_outputs_convert = [post_pred(i) for i in val_outputs_list]
                val_labels_convert = [post_label(i) for i in val_labels_list]
                
                dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)

            metric = dice_metric.aggregate().item()
            current_val_metric = metric
            dice_metric.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "/workspace/unetplusplus_best_metric_model.pth")
                print(">>> Saved new best metric model")

            print(f"Current mean dice: {metric:.4f}")
            print(f"Best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
            
    with open(log_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_loss, current_val_metric, best_metric, current_lr])

print(f"Training completed. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
