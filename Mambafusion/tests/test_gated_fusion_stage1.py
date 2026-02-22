import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'pcdet/models/backbones_2d/fuser/gated_fusion.py'
spec = importlib.util.spec_from_file_location('gated_fusion', MODULE_PATH)
gated_fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gated_fusion)

ModalityDropout = gated_fusion.ModalityDropout
AlignmentProxy = gated_fusion.AlignmentProxy
GlobalGatedFusion = gated_fusion.GlobalGatedFusion


def test_shape_consistency_with_baseline_concat():
    b, c1, c2, h, w = 2, 80, 128, 8, 8
    zc = torch.randn(b, c1, h, w)
    zl = torch.randn(b, c2, h, w)

    gate = GlobalGatedFusion(hidden_dim=32, use_alignment_proxy=False)
    zf, alpha = gate(zc, zl)

    assert zf.shape == torch.cat([zc, zl], dim=1).shape
    assert alpha.shape == (b, 2)


def test_modality_dropout_stats_and_non_empty():
    torch.manual_seed(0)
    b, h, w = 64, 4, 4
    zc = torch.ones(b, 4, h, w)
    zl = torch.ones(b, 6, h, w)

    md = ModalityDropout(p_cam=0.3, p_lidar=0.2)
    md.train()

    masks = []
    for _ in range(200):
        _, _, m = md(zc, zl)
        masks.append(m)
    masks = torch.cat(masks, dim=0)

    assert torch.all(masks.sum(dim=1) >= 1.0)
    keep_cam = masks[:, 0].mean().item()
    keep_lidar = masks[:, 1].mean().item()
    assert abs(keep_cam - 0.7) < 0.1
    assert abs(keep_lidar - 0.8) < 0.1


def test_gate_degenerate_matches_single_modality():
    b, c1, c2, h, w = 2, 3, 5, 4, 4
    zc = torch.randn(b, c1, h, w)
    zl = torch.randn(b, c2, h, w)

    out_cam = torch.cat([zc, torch.zeros_like(zl)], dim=1)
    out_lidar = torch.cat([torch.zeros_like(zc), zl], dim=1)

    alpha_cam = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    alpha_lidar = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    fused_cam = torch.cat([
        zc * alpha_cam[:, 0].view(-1, 1, 1, 1),
        zl * alpha_cam[:, 1].view(-1, 1, 1, 1),
    ], dim=1)
    fused_lidar = torch.cat([
        zc * alpha_lidar[:, 0].view(-1, 1, 1, 1),
        zl * alpha_lidar[:, 1].view(-1, 1, 1, 1),
    ], dim=1)

    assert torch.allclose(fused_cam, out_cam)
    assert torch.allclose(fused_lidar, out_lidar)


def test_alignment_proxy_cosine_range_and_none_zero():
    b, h, w = 3, 6, 6
    zc = torch.randn(b, 8, h, w)
    zl = torch.randn(b, 10, h, w)

    proxy = AlignmentProxy(mode='feat_cosine', proj_dim=16)
    q = proxy(zc, zl)
    assert q.shape == (b, 3)
    assert torch.all(q[:, 0] <= 1.0 + 1e-5)
    assert torch.all(q[:, 0] >= -1.0 - 1e-5)

    proxy_none = AlignmentProxy(mode='none', proj_dim=16)
    q0 = proxy_none(zc, zl)
    assert torch.allclose(q0, torch.zeros_like(q0))