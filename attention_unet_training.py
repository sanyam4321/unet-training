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

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.upconv = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        theta_x = self.W_x(x)
        phi_g = self.W_g(g)
        f = self.relu(theta_x + phi_g)
        psi_f = self.psi(f)
        upsampled_psi = self.upconv(psi_f)
        alpha = self.sigmoid(upsampled_psi)
        return x * alpha


class AttentionUNET3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):  # Adjusted default for Colon (Background + Tumor)
        super(AttentionUNET3D, self).__init__()
        feature_maps = [64, 128, 256, 512, 1024]
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv1 = DoubleConv(in_channels, feature_maps[0])
        self.conv2 = DoubleConv(feature_maps[0], feature_maps[1])
        self.conv3 = DoubleConv(feature_maps[1], feature_maps[2])
        self.conv4 = DoubleConv(feature_maps[2], feature_maps[3])

        self.bottleneck = DoubleConv(feature_maps[3], feature_maps[4])

        self.att4 = AttentionGate3D(F_g=feature_maps[4], F_l=feature_maps[3], F_int=feature_maps[3] // 2)
        self.up_conv4 = nn.ConvTranspose3d(feature_maps[4], feature_maps[3], kernel_size=2, stride=2)
        self.conv4_r = DoubleConv(2 * feature_maps[3], feature_maps[3])

        self.att3 = AttentionGate3D(F_g=feature_maps[3], F_l=feature_maps[2], F_int=feature_maps[2] // 2)
        self.up_conv3 = nn.ConvTranspose3d(feature_maps[3], feature_maps[2], kernel_size=2, stride=2)
        self.conv3_r = DoubleConv(2 * feature_maps[2], feature_maps[2])

        self.att2 = AttentionGate3D(F_g=feature_maps[2], F_l=feature_maps[1], F_int=feature_maps[1] // 2)
        self.up_conv2 = nn.ConvTranspose3d(feature_maps[2], feature_maps[1], kernel_size=2, stride=2)
        self.conv2_r = DoubleConv(2 * feature_maps[1], feature_maps[1])

        self.att1 = AttentionGate3D(F_g=feature_maps[1], F_l=feature_maps[0], F_int=feature_maps[0] // 2)
        self.up_conv1 = nn.ConvTranspose3d(feature_maps[1], feature_maps[0], kernel_size=2, stride=2)
        self.conv1_r = DoubleConv(2 * feature_maps[0], feature_maps[0])

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

        at_s4 = self.att4(g=bt, x=s4)
        up4 = self.up_conv4(bt)
        r_s4 = self.conv4_r(torch.cat((at_s4, up4), dim=1))

        at_s3 = self.att3(g=r_s4, x=s3)
        up3 = self.up_conv3(r_s4)
        r_s3 = self.conv3_r(torch.cat((at_s3, up3), dim=1))

        at_s2 = self.att2(g=r_s3, x=s2)
        up2 = self.up_conv2(r_s3)
        r_s2 = self.conv2_r(torch.cat((at_s2, up2), dim=1))

        at_s1 = self.att1(g=r_s2, x=s1)
        up1 = self.up_conv1(r_s2)
        r_s1 = self.conv1_r(torch.cat((at_s1, up1), dim=1))

        return self.segments(r_s1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNET3D(in_channels=1, out_channels=2).to(device)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

data_dir = "/mnt/ramdisk/Task10_Colon_Preprocessed"
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "colon*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "colon*.nii.gz")))

print(f"images: {len(train_images)}")

data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

train_files, val_files = data_dicts[:-10], data_dicts[-10:]

set_determinism(seed=0)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(128, 128, 64),
        pos=1,
        neg=1,
        num_samples=10,
        image_key="image",
        image_threshold=0,
    ),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)


max_epochs = 100
warmup_epochs = 10

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

val_interval = 2
best_metric = -1
best_metric_epoch = -1

import csv

log_filename = "/workspace/attention_unet_training_log.csv"
with open(log_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_dice", "best_dice"])

print(f"Logging metrics to {log_filename}")

scaler = torch.amp.GradScaler(device="cuda")


for epoch in range(max_epochs):
    print(f"-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        
    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    current_val_metric = ""
    if (epoch + 1) % val_interval == 0:
        print(f"-----------------Validation-----------------")
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                roi_size = (128, 128, 64)
                sw_batch_size = 4
                
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

                val_outputs = [torch.argmax(i, dim=0, keepdim=True) for i in decollate_batch(val_outputs)]
                val_labels = [i for i in decollate_batch(val_labels)]

                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            scheduler.step(metric)
            current_val_metric = metric
            dice_metric.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "/workspace/attention_unet_best_metric_model.pth")
                print("saved new best metric model")

            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}")
            print(f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
    with open(log_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_loss, current_val_metric, best_metric])

print(f"Training completed. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
