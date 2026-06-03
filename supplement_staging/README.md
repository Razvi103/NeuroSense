# MIDAS: Multi-Layer Invariant Domain Adaptation for Cross-Patient Seizure Detection

MIDAS adds **channel-attention aggregation** and **multi-layer gradient-reversal
patient discriminators** to a pre-trained
[LaBraM](https://github.com/935963004/LaBraM) backbone for cross-patient
seizure detection on scalp EEG.

## Upstream files not included

The following unmodified files from
[LaBraM](https://github.com/935963004/LaBraM) are required but not
shipped in this supplement (they are identical to upstream):

- `LaBraM/optim_factory.py`
- `LaBraM/data_processor/dataset.py`
- `LaBraM/data_processor/data_preprocess.py`

Download them from the LaBraM repository and place them in the corresponding
directories.

## Installation

```bash
pip install -r requirements.txt
```

The pre-trained LaBraM checkpoint (`labram-base.pth`) can be obtained from the
[LaBraM repository](https://github.com/935963004/LaBraM).

## Data preparation

### CHB-MIT (PhysioNet, open access)

Download from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

```bash
# per-patient H5 files for LOPOCV
python LaBraM/dataset_maker/adapt_chbmit_per_patient.py \
    --data_root /path/to/CHB-MIT_Raw \
    --output_dir ./data/CHBMIT_per_patient
```

### TUSZ (requires IISP data use agreement)

Obtain from the [TUH EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/).

```bash
python LaBraM/dataset_maker/make_TUSZ.py \
    --data_root /path/to/tusz_v2.0.3 \
    --output_dir ./data/TUSZ
```

## Training

### Single split (TUSZ or CHB-MIT global split)

```bash
python LaBraM/run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --data_path ./data/TUSZ \
    --finetune /path/to/labram-base.pth \
    --dataset TUSZ \
    --adversarial \
    --adv_lambda 0.01 \
    --adv_gamma 5.0 \
    --intermediate_layers "3,7" \
    --epochs 30 --lr 3e-6 --batch_size 512
```

### Leave-One-Patient-Out Cross-Validation (CHB-MIT)

```bash
python LaBraM/run_lopocv_chbmit.py \
    --data_dir ./data/CHBMIT_per_patient \
    --finetune /path/to/labram-base.pth \
    --output_dir ./results/lopocv \
    --epochs 10 --lr 3e-6 --batch_size 1024
```

## Evaluation

```bash
# single checkpoint evaluation
python LaBraM/evaluate_checkpoint.py \
    --data_path ./data/CHBMIT \
    --checkpoint ./checkpoints/checkpoint-best.pth \
    --adversarial --dataset CHBMIT

# aggregate LOPOCV results
python LaBraM/eval_lopocv_full.py \
    --lopocv_dir ./results/lopocv \
    --data_dir ./data/CHBMIT_per_patient

# post-processing hyperparameter tuning
python LaBraM/tune_and_plot.py \
    --data_path ./data/CHBMIT \
    --checkpoint ./checkpoints/checkpoint-best.pth \
    --adversarial
```

## License

The upstream LaBraM code is released under the MIT License (see `LaBraM/LICENSE`).
All modifications in this supplement are provided under the same license.
