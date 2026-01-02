import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim import Adam

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


class UNET3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNET3D, self).__init__()

        feature_maps = [64, 128, 256, 512, 1024]
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv1 = DoubleConv(in_channels, feature_maps[0])
        self.conv2 = DoubleConv(feature_maps[0], feature_maps[1])
        self.conv3 = DoubleConv(feature_maps[1], feature_maps[2])
        self.conv4 = DoubleConv(feature_maps[2], feature_maps[3])

        self.bottleneck = DoubleConv(feature_maps[3], feature_maps[4])

        self.up_conv1 = nn.ConvTranspose3d(feature_maps[4], feature_maps[3], kernel_size=2, stride=2)
        self.conv1_r = DoubleConv(2 * feature_maps[3], feature_maps[3])

        self.up_conv2 = nn.ConvTranspose3d(feature_maps[3], feature_maps[2], kernel_size=2, stride=2)
        self.conv2_r = DoubleConv(2 * feature_maps[2], feature_maps[2])

        self.up_conv3 = nn.ConvTranspose3d(feature_maps[2], feature_maps[1], kernel_size=2, stride=2)
        self.conv3_r = DoubleConv(2 * feature_maps[1], feature_maps[1])

        self.up_conv4 = nn.ConvTranspose3d(feature_maps[1], feature_maps[0], kernel_size=2, stride=2)
        self.conv4_r = DoubleConv(2 * feature_maps[0], feature_maps[0])

        self.segments = nn.Conv3d(feature_maps[0], out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.conv1(x)
        mp1 = self.pool(s1)

        s2 = self.conv2(mp1)
        mp2 = self.pool(s2)

        s3 = self.conv3(mp2)
        mp3 = self.pool(s3)

        s4 = self.conv4(mp3)
        mp4 = self.pool(s4)

        bt = self.bottleneck(mp4)

        up1 = self.up_conv1(bt)
        r_s4 = self.conv1_r(torch.cat((s4, up1), dim=1))

        up2 = self.up_conv2(r_s4)
        r_s3 = self.conv2_r(torch.cat((s3, up2), dim=1))

        up3 = self.up_conv3(r_s3)
        r_s2 = self.conv3_r(torch.cat((s2, up3), dim=1))

        up4 = self.up_conv4(r_s2)
        r_s1 = self.conv4_r(torch.cat((s1, up4), dim=1))

        return self.segments(r_s1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET3D(in_channels=1, out_channels=3).to(device)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

import os
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
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
)
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.utils import set_determinism

import os
import glob
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped
)

proc_dir = "/home/Task07_Pancreas"
train_images = sorted(glob.glob(os.path.join(proc_dir, "imagesTr", "pancreas*.nii")))
train_labels = sorted(glob.glob(os.path.join(proc_dir, "labelsTr", "pancreas*.nii")))

print(f"total images: {len(train_images)}, total labels: {len(train_labels)}")

data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]

val_split_idx = int(len(data_dicts) * 0.9)
train_files, val_files = data_dicts[:val_split_idx], data_dicts[val_split_idx:]

print(f"train size: {len(train_files)}, validation size: {len(val_files)}")

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

train_ds = CacheDataset(
    data=train_files, 
    transform=train_transforms, 
    cache_rate=0.1, 
    num_workers=4
)

train_loader = DataLoader(
    train_ds, 
    batch_size=5, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)

val_ds = CacheDataset(
    data=val_files, 
    transform=val_transforms, 
    cache_rate=0.1,
    num_workers=4
)

val_loader = DataLoader(
    val_ds, 
    batch_size=1, 
    shuffle=False, 
    num_workers=4
)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

import csv
import torch
from tqdm import tqdm

max_epochs = 300
accumulation_steps = 5
val_interval = 5
best_metric = -1
best_metric_epoch = -1

log_filename = "/workspace/unet_training_log_run2.5.csv"
with open(log_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_dice", "best_dice"])

print(f"Logging metrics to {log_filename}")

scaler = torch.amp.GradScaler(device="cuda")

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
            
            # --- GRADIENT ACCUMULATION LOGIC ---
            # Normalize loss so gradients are averaged, not summed
            loss = loss / accumulation_steps 
        
        # Accumulate gradients (PyTorch adds them by default)
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 

        current_loss = loss.item() * accumulation_steps
        epoch_loss += current_loss
        
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

    if (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss /= step
    print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
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
                
                
                val_outputs = [torch.argmax(i, dim=0, keepdim=True) for i in decollate_batch(val_outputs)]
                val_labels = [i for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            current_val_metric = metric
            dice_metric.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "/workspace/unet_baseline_best_metric_model_run2.5.pth")
                print(">>> Saved new best metric model")

            print(f"Current mean dice: {metric:.4f}")
            print(f"Best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
            
    with open(log_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_loss, current_val_metric, best_metric])

print(f"Training completed. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
