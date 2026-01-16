# NeuroSense: EEG-Based Seizure Detection

Deep learning framework for automated seizure detection using the Large Brain Model (LaBraM) architecture, fine-tuned on the CHB-MIT Scalp EEG Database.

## Overview

This project implements a performant seizure detection system using transformer-based neural networks. The system is built on LaBraM, a foundation model pre-trained on over 2,500 hours of diverse EEG data, and fine-tuned specifically for epileptic seizure detection.

### Key Features

- Pre-trained transformer architecture adapted for seizure detection
- Automated preprocessing pipeline for CHB-MIT dataset
- Advanced post-processing with dual-threshold detection and temporal smoothing
- Comprehensive evaluation metrics including epoch-based analysis
- Docker support for reproducible environments
- Professional visualization tools for seizure detection analysis

## Project Structure

```
NeuroSense/
├── LaBraM/
│   ├── dataset_maker/          # Data preprocessing scripts
│   │   ├── adapt_chbmit.py     # CHB-MIT dataset adapter
│   │   ├── dataset_chbmit.py   # PyTorch dataset class
│   │   ├── make_h5dataset_for_pretrain.py
│   │   ├── make_TUAB.py
│   │   └── make_TUEV.py
│   ├── checkpoints/            # Model weights
│   │   ├── labram-base.pth     # Pre-trained weights
│   │   └── finetune_chbmit_v1/ # Fine-tuned models
│   ├── data_processor/         # Data processing utilities
│   ├── engine_for_finetuning.py
│   ├── engine_for_pretraining.py
│   ├── modeling_finetune.py
│   ├── modeling_pretrain.py
│   ├── run_class_finetuning.py # Training script
│   ├── evaluate_checkpoint.py  # Main evaluation script
│   ├── eval_szcore_metrics.py  # Epoch-based metrics
│   ├── tune_and_plot.py        # Hyperparameter tuning
│   ├── visualize_predictions.py # Seizure visualization
│   └── utils.py
├── Dockerfile
├── .dockerignore
└── README.md
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 11.8 support
- Docker (recommended) or Python 3.11+
- CHB-MIT dataset (available from PhysioNet)

### Installation

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
cd /path/to/NeuroSense

# Build the Docker image
docker build -t neurosense/labram:latest .

# Run the container
docker run --gpus all -it --rm \
  -v $(pwd)/LaBraM:/workspace/LaBraM \
  -v /path/to/datasets:/workspace/LaBraM/datasets \
  -p 6006:6006 \
  neurosense/labram:latest
```

#### Option 2: Local Installation

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

### CHB-MIT Dataset

1. Download the CHB-MIT Scalp EEG Database from PhysioNet:
   ```bash
   wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
   ```

2. Preprocess the data using the provided script:
   ```bash
   cd LaBraM/dataset_maker
   python adapt_chbmit.py
   ```

The preprocessing pipeline performs:
- Channel selection and mapping to standard 10-20 system (23 channels)
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

# Single GPU training
python run_class_finetuning.py \
  --data_path ../datasets/CHBMIT \
  --output_dir ./checkpoints/finetune_chbmit \
  --log_dir ./log/finetune_chbmit \
  --finetune ./checkpoints/labram-base.pth \
  --dataset CHBMIT \
  --batch_size 64 \
  --lr 5e-4 \
  --epochs 30 \
  --warmup_epochs 5

# Multi-GPU training (recommended)
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

This script:
- Generates ROC and Precision-Recall curves
- Performs grid search over post-processing parameters
- Reports best configurations based on Cohen's Kappa and F1 score
- Saves results to `metrics_curves.png`

### Visualization

Generate seizure detection visualizations:

```bash
python visualize_predictions.py \
  --data_path ../datasets/CHBMIT \
  --checkpoint ./checkpoints/finetune_chbmit_v1/checkpoint-19.pth \
  --t_high 0.4 \
  --t_low 0.2 \
  --smooth 5 \
  --min_dur 5
```

Outputs:
- `seizure_viz_1.png` through `seizure_viz_5.png`
- Each visualization shows ground truth, model probabilities, smoothed predictions, and final detection

### Epoch-based Evaluation

Perform rigorous epoch-based evaluation (10-second windows):

```bash
python eval_szcore_metrics.py \
  --data_path ../datasets/CHBMIT \
  --checkpoint ./checkpoints/finetune_chbmit_v1/checkpoint-9.pth
```

## Evaluation Metrics

The system provides multiple evaluation perspectives:

### Point-wise Metrics (Sample-level)
- **Sensitivity (Recall)**: Proportion of seizure samples correctly identified
- **Specificity**: Proportion of non-seizure samples correctly identified
- **Precision (PPV)**: Proportion of predicted seizures that are actual seizures
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: Area under the ROC curve
- **AUPRC**: Area under the Precision-Recall curve (most important for imbalanced data)

### Event-based Metrics (Seizure-level)
- **Detection Rate**: Percentage of seizure events detected
- **False Alarm Rate**: Number of false alarms per hour
- **Event Precision**: Proportion of predicted events that overlap with true events
- **Event F1**: Harmonic mean of event-level precision and recall

### Epoch-based Metrics (10-second windows)
- **Epoch F1 Score**: F1 computed on 10-second epochs
- **Cohen's Kappa**: Inter-rater agreement metric (>0.4 indicates strong performance)
- **Epoch Sensitivity/Specificity**: Performance on temporal epochs

## Model Architecture

The system uses a transformer-based architecture with the following specifications:

### Base Model
- **Architecture**: labram_base_patch200_200
- **Patch Size**: 200 samples (1 second at 200 Hz)
- **Input Channels**: 23 EEG channels mapped to standard 10-20 system
- **Output**: Binary classification (seizure/non-seizure)
- **Loss Function**: Weighted Binary Cross-Entropy (pos_weight=200.0)

### Channel Mapping
The model expects 23 channels in the following order:
```
FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ, CZ-PZ, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8
```

### Post-processing Pipeline

1. **Temporal Smoothing**: Moving average window to reduce noise
2. **Dual-threshold Detection**: 
   - High threshold for seizure onset
   - Low threshold for seizure continuation (hysteresis)
3. **Duration Filtering**: Remove detections shorter than minimum duration

## Performance

### CHB-MIT Dataset Results
Best checkpoint (checkpoint-19.pth) achieves:
- AUPRC: >0.XX (reported after tuning)
- Sensitivity: >XX%
- Specificity: >XX%
- False Alarm Rate: <X per hour
- Cohen's Kappa: >0.4

## Advanced Usage

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

Access at: http://localhost:6006

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 32`
   - Use gradient accumulation: `--update_freq 2`

2. **Missing Channels in Dataset**
   - Check channel mapping in `dataset_maker/adapt_chbmit.py`
   - Ensure raw EEG files have all required channels

3. **Poor Performance**
   - Verify data normalization (should be divided by 100 in engine, not dataset)
   - Tune learning rate and warmup epochs
   - Adjust post-processing thresholds using `tune_and_plot.py`

## Citation

If you use this code or the LaBraM model, please cite:

```bibtex
@inproceedings{jiang2024large,
  title={Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI},
  author={Wei-Bang Jiang and Li-Ming Zhao and Bao-Liang Lu},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=QzTpTRVtrP}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- Original LaBraM implementation by Wei-Bang Jiang et al.
- CHB-MIT dataset provided by Boston Children's Hospital
- PyTorch and timm libraries for deep learning infrastructure

## Contact

For questions or issues, please open an issue on the project repository.

---

**Note**: This is a research project for seizure detection. The system is not intended for clinical use without proper validation and regulatory approval.
