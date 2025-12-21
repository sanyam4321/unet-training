#!/bin/bash

# 1. Copy the dataset to /home
# echo "--- Copying Dataset ---"
# if [ -d "/home/Task10_Colon_Preprocessed" ]; then
#     echo "Dataset already exists in /home, skipping copy."
# else
#     # Using -r for recursive and -p to preserve timestamps/permissions
#     cp -rp /workspace/Task10_Colon_Preprocessed /home/
#     echo "Dataset copied successfully."
# fi

# 1. Create a folder for the mount point
sudo mkdir -p /mnt/ramdisk

sudo mount -t tmpfs -o size=40G tmpfs /mnt/ramdisk

sudo rsync -av --progress /workspace/Task10_Colon_Preprocessed/ /mnt/ramdisk/

sudo chmod -R 777 /mnt/ramdisk

# 2. Install nano and essential libraries
echo "--- Installing Nano and System Dependencies ---"
apt update && apt install -y nano

# 4. Install requirements
echo "--- Installing Python Requirements ---"
if [ -f "requirements.txt" ]; then
    # We add nibabel and tqdm explicitly just in case they aren't in your requirements.txt
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install nibabel tqdm
else
    echo "requirements.txt not found! Installing standard MONAI stack..."
    pip install monai[nibabel,tqdm] torch torchvision torchaudio
fi

echo "--- Setup Complete! ---"
echo "You are now in: $(pwd)"
