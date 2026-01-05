🧠 Benchmarking 3D U-Net Variants on the Medical Decathlon Pancreas Dataset
Overview

This project presents a systematic benchmarking and optimization study of multiple 3D U-Net–based architectures for pancreas segmentation on the Medical Segmentation Decathlon (MSD) – Task 07: Pancreas dataset.

The work focuses not only on model accuracy but also on high-performance training engineering, addressing common bottlenecks in large-scale 3D medical image segmentation such as I/O latency, GPU under-utilization, and VRAM constraints.

Key Contributions
🚀 Performance Engineering

50% training throughput improvement using:

Mixed Precision Training (AMP / BFloat16)

Optimized memory access patterns

80% GPU utilization increase by:

Eliminating I/O bottlenecks

Implementing an offline multi-threaded preprocessing pipeline for NIfTI volumes

Scalable multi-GPU training using:

Gradient Accumulation

DistributedDataParallel (DDP) across 2× NVIDIA A100 GPUs

Enables training on large 3D volumes beyond single-GPU VRAM limits

🧬 End-to-End 3D Segmentation Pipeline

Native NIfTI (.nii.gz) data handling

Preprocessing:

Resampling to uniform voxel spacing

Hounsfield Unit (HU) clipping

Intensity normalization

Data augmentation (3D):

Rotation

Scaling

Elastic deformation

Fully modular and extensible training pipeline

Model Architectures Implemented

All models were implemented from scratch and trained on GPUs (P100 / A100):

Model	Description
3D U-Net	Baseline encoder–decoder architecture
3D ResUNet	Residual connections for improved gradient flow
Attention-Gated U-Net	Attention gates to suppress irrelevant regions
Attention-Gated ResUNet	Combines residual learning with attention mechanisms
Training Details

Loss Function: Dice Loss + Cross-Entropy Loss

Optimizer: Adam

Learning Rate Scheduler: Cosine Annealing

Precision: FP16 / BF16 (AMP)

Distributed Training: PyTorch DDP

Results

A rigorous comparative evaluation was conducted across all architectures.

Model	Dice Similarity Coefficient (DSC)
3D U-Net (Baseline)	Lower
3D ResUNet	Improved
Attention U-Net	Further improvement
Attention-Gated ResUNet	0.67 (Best)

The Attention-Gated ResUNet demonstrated superior boundary delineation and robustness compared to the baseline U-Net on the Pancreas dataset.

Getting Started
1️⃣ Clone the Repository
git clone https://github.com/sanyam9922/unet-training.git
cd your-repo-name

2️⃣ Upload Dataset to Remote GPU Server

Copy the Medical Decathlon Pancreas dataset to the remote machine:

scp -r /local/path/to/MSD_Pancreas user@remote_server:/remote/path/data/

3️⃣ Set Up the Environment
bash setup_env.sh


This script installs required Python dependencies, CUDA-compatible PyTorch, and auxiliary libraries.

4️⃣ Run Training

Train different models by executing the corresponding scripts:

python training/unet_training.py


or for multi-GPU training:

torchrun --nproc_per_node=2 unet_training_ddp.py

Dataset

Medical Segmentation Decathlon (MSD)

Task 07: Pancreas

Dataset contains contrast-enhanced abdominal CT scans with voxel-level pancreas annotations.

⚠️ Dataset is not included due to licensing restrictions.

Hardware

NVIDIA P100 (development & baseline training)

NVIDIA A100 ×2 (distributed training & performance optimization)

Future Work

Transformer-based 3D segmentation (UNETR, Swin-UNet)

Self-supervised pretraining for volumetric medical data

Inference optimization (TensorRT, ONNX)

Author

Sanyam
2025

If you find this project useful, feel free to ⭐ the repository.
