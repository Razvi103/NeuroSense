# NeuroSense: EEG-Based Seizure Detection

Deep learning framework for automated seizure detection using the Large Brain Model (LaBraM) architecture, fine-tuned on the CHB-MIT Scalp EEG Database.

## Overview

This project implements a performant seizure detection system using transformer-based neural networks. The system is built on LaBraM, a foundation model pre-trained on over 2,500 hours of diverse EEG data, and fine-tuned specifically for epileptic seizure detection.

### Key Features

- Pre-trained transformer architecture adapted for seizure detection
- Automated preprocessing pipeline for CHB-MIT dataset
- Advanced post-processing with dual-threshold detection and temporal smoothing
- Comprehensive evaluation metrics including epoch-based analysis
- Professional visualization tools for seizure detection analysis

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 11.8 support
- CHB-MIT dataset (available from PhysioNet) and TUSZ dataset



#### Local Installation

```bash
# Create conda environment
conda create -n labram python=3.11
conda activate labram

# Install PyTorch with CUDA support
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
cd LaBraM
pip install tensorboardX
pip install -r requirements.txt
```

## Data Preparation

The preprocessing pipeline performs:
- Channel selection and mapping to standard 10-20 system (23 channels for CHB-MIT and 19 channels for TUSZ)
- Notch filtering at 60 Hz to remove power line noise
- Bandpass filtering between 0.1-75 Hz
- Resampling to 200 Hz for consistency
- Segmentation into 2-second windows with 1-second stride
- Subject-independent train/validation/test split (80/10/10)

Output format: HDF5 files containing preprocessed EEG segments and labels

## Usage

### Training

Fine-tune the pre-trained LaBraM model on CHB-MIT dataset:

```bash
cd LaBraM

# single gpu training
python run_class_finetuning.py \
  --data_path ../datasets/CHBMIT \
  --output_dir ./checkpoints/finetune_chbmit \
  --log_dir ./log/finetune_chbmit \
  --finetune ./checkpoints/labram-base.pth \
  --dataset CHBMIT \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 30 \
  --warmup_epochs 5

# multi-gpu trainin
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 \
  run_class_finetuning.py \
  --data_path ../datasets/CHBMIT \
  --output_dir ./checkpoints/finetune_chbmit \
  --finetune ./checkpoints/labram-base.pth \
  --dataset CHBMIT \
  --batch_size 64 \
  --lr 5e-4 \
  --epochs 30
```

### Evaluation

Evaluate a trained checkpoint with comprehensive metrics:

```bash
python evaluate_checkpoint.py \
  --data_path ../datasets/CHBMIT \
  --checkpoint ./checkpoints/finetune_chbmit_v1/checkpoint-19.pth \
  --batch_size 2048 \
  --t_high 0.4 \
  --t_low 0.2 \
  --smooth 5 \
  --min_dur 5
```

Parameters:
- `t_high`: High threshold for seizure onset detection (default: 0.4)
- `t_low`: Low threshold for seizure continuation (default: 0.2)
- `smooth`: Temporal smoothing window in seconds (default: 5)
- `min_dur`: Minimum seizure duration in seconds (default: 5)

### Hyperparameter Tuning

Optimize post-processing parameters using grid search:

```bash
python tune_and_plot.py \
  --data_path ../datasets/CHBMIT \
  --checkpoint ./checkpoints/finetune_chbmit_v1/checkpoint-19.pth
```


## Model Architecture

The system uses a transformer-based architecture with the following specifications:


### Training from Scratch

If you want to pre-train LaBraM from scratch:

1. Train the neural tokenizer (VQ-NSP):
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 \
  run_vqnsp_training.py \
  --output_dir ./checkpoints/vqnsp/ \
  --model vqnsp_encoder_base_decoder_3x200x12 \
  --batch_size 128 \
  --epochs 100
```

2. Pre-train LaBraM:
```bash
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 \
  run_labram_pretraining.py \
  --output_dir ./checkpoints/labram_base \
  --tokenizer_weight ./checkpoints/vqnsp.pth \
  --batch_size 64 \
  --epochs 50
```

### TensorBoard Monitoring

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=./LaBraM/log --port 6006
```

