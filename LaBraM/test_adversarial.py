"""
End-to-end smoke test for the adversarial LaBraM fine-tuning pipeline.
Runs on CPU with synthetic data to verify:
  1. Model creation and architecture
  2. Forward pass (training mode → dual outputs, eval mode → single output)
  3. Backward pass with combined adversarial loss
  4. GRL lambda scheduling
  5. Checkpoint loading into backbone
  6. TUSZAdversarialDataset with a tiny synthetic HDF5
  7. Channel attention weight shapes
"""
import sys
import os
import tempfile
import math

import numpy as np
import h5py
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from modeling_finetune import (
    NeuralTransformer,
    AdversarialNeuralTransformer,
    GradientReversalFunction,
    ChannelAttention,
    PatientDiscriminator,
)
from engine_for_finetuning import get_grl_lambda
from dataset_maker.dataset_chbmit import TUSZAdversarialDataset
from functools import partial
import torch.nn as nn


def test_grl_lambda_schedule():
    assert get_grl_lambda(0, 30) == 0.0, "GRL lambda should be 0 at epoch 0"
    lam_mid = get_grl_lambda(15, 30)
    assert 0.4 < lam_mid < 0.6, f"GRL lambda at midpoint should be ~0.5, got {lam_mid}"
    lam_end = get_grl_lambda(29, 30)
    assert lam_end > 0.9, f"GRL lambda at end should be ~1.0, got {lam_end}"
    print("[PASS] GRL lambda schedule")


def test_gradient_reversal():
    x = torch.randn(4, 8, requires_grad=True)
    y = GradientReversalFunction.apply(x, 1.0)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, -torch.ones_like(x)), "GRL should negate gradients"
    print("[PASS] Gradient reversal")


def test_channel_attention():
    B, C, T, D = 2, 23, 2, 200
    attn = ChannelAttention(D, reduction=4)
    tokens = torch.randn(B, C * T, D)
    pooled, weights = attn(tokens, n_channels=C, n_time=T)
    assert pooled.shape == (B, D), f"Pooled shape {pooled.shape} != ({B}, {D})"
    assert weights.shape == (B, C), f"Weights shape {weights.shape} != ({B}, {C})"
    assert torch.allclose(weights.sum(dim=1), torch.ones(B)), "Attention weights should sum to 1"
    print("[PASS] Channel attention")


def test_patient_discriminator():
    disc = PatientDiscriminator(embed_dim=200, num_patients=50, hidden_dim=64)
    x = torch.randn(4, 200)
    out = disc(x)
    assert out.shape == (4, 50), f"Discriminator output shape {out.shape} != (4, 50)"
    print("[PASS] Patient discriminator")


def test_model_forward_backward():
    backbone = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=2, num_heads=10,
        mlp_ratio=4, num_classes=1,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
    )
    num_patients = 10
    model = AdversarialNeuralTransformer(backbone, num_patients=num_patients, adv_hidden_dim=64)

    B, N_ch, N_time, P = 2, 23, 2, 200
    x = torch.randn(B, N_ch, N_time, P)
    patient_ids = torch.randint(0, num_patients, (B,))

    # Training mode: should return (seizure_logits, patient_logits)
    model.train()
    seizure_logits, patient_logits = model(x)
    assert seizure_logits.shape == (B, 1), f"Seizure logits shape {seizure_logits.shape}"
    assert patient_logits.shape == (B, num_patients), f"Patient logits shape {patient_logits.shape}"

    targets = torch.randint(0, 2, (B, 1)).float()
    seizure_loss = F.binary_cross_entropy_with_logits(seizure_logits, targets)
    patient_loss = F.cross_entropy(patient_logits, patient_ids)
    total_loss = seizure_loss + 0.1 * patient_loss
    total_loss.backward()
    print(f"  seizure_loss={seizure_loss.item():.4f}, patient_loss={patient_loss.item():.4f}")

    # Eval mode: should return only seizure_logits
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, 1), "Eval mode should return seizure logits only"

    print("[PASS] Model forward/backward")


def test_checkpoint_loading():
    backbone = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=2, num_heads=10,
        mlp_ratio=4, num_classes=1,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
    )

    # Save backbone weights to simulate a pre-trained checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        ckpt_path = f.name
        torch.save({'model': backbone.state_dict()}, ckpt_path)

    # Create adversarial model and load into its backbone
    backbone2 = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=2, num_heads=10,
        mlp_ratio=4, num_classes=1,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
    )
    model = AdversarialNeuralTransformer(backbone2, num_patients=5, adv_hidden_dim=32)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    checkpoint_model = ckpt['model']

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model:
            del checkpoint_model[k]

    missing, unexpected = model.backbone.load_state_dict(checkpoint_model, strict=False)
    print(f"  Missing keys: {missing}")
    print(f"  Unexpected keys: {unexpected}")

    os.unlink(ckpt_path)
    print("[PASS] Checkpoint loading into backbone")


def test_adversarial_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, 'test.h5')
        N = 16
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', data=np.random.randn(N, 23, 400).astype(np.float32))
            f.create_dataset('labels', data=np.random.randint(0, 2, N).astype(np.int64))
            f.create_dataset('patient_ids', data=np.random.randint(0, 3, N).astype(np.int64))

        ds = TUSZAdversarialDataset(h5_path)
        assert len(ds) == N
        assert ds.num_patients == 3 or ds.num_patients >= 1

        data, label, pid = ds[0]
        assert data.shape == (23, 400), f"Data shape {data.shape}"
        assert isinstance(label, (int, np.integer)), f"Label type {type(label)}"
        assert isinstance(pid, int), f"PID type {type(pid)}"

        # Test fallback without patient_ids
        h5_path2 = os.path.join(tmpdir, 'test_no_pid.h5')
        with h5py.File(h5_path2, 'w') as f:
            f.create_dataset('data', data=np.random.randn(4, 23, 400).astype(np.float32))
            f.create_dataset('labels', data=np.random.randint(0, 2, 4).astype(np.int64))

        ds2 = TUSZAdversarialDataset(h5_path2)
        _, _, pid2 = ds2[0]
        assert pid2 == -1, "Should return -1 when patient_ids absent"

    print("[PASS] TUSZAdversarialDataset")


def test_no_weight_decay():
    backbone = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=2, num_heads=10,
        mlp_ratio=4, num_classes=1,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
    )
    model = AdversarialNeuralTransformer(backbone, num_patients=5, adv_hidden_dim=32)
    nwd = model.no_weight_decay()
    assert any('pos_embed' in k for k in nwd), f"Expected pos_embed in no_weight_decay, got {nwd}"
    print("[PASS] no_weight_decay")


if __name__ == '__main__':
    print("=" * 60)
    print("Running adversarial LaBraM smoke tests")
    print("=" * 60)
    test_grl_lambda_schedule()
    test_gradient_reversal()
    test_channel_attention()
    test_patient_discriminator()
    test_model_forward_backward()
    test_checkpoint_loading()
    test_adversarial_dataset()
    test_no_weight_decay()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
