#!/bin/bash


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
