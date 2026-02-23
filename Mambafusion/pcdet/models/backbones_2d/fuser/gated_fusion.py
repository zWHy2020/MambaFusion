import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ModalityDropout(nn.Module):
    """Structured modality dropout for camera/lidar BEV features."""

    def __init__(self, p_cam: float = 0.0, p_lidar: float = 0.0):
        super().__init__()
        self.p_cam = float(p_cam)
        self.p_lidar = float(p_lidar)

    def forward(self, z_cam: torch.Tensor, z_lidar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = z_cam.shape[0]
        device = z_cam.device

        if (not self.training) or (self.p_cam <= 0 and self.p_lidar <= 0):
            mask_cam = torch.ones(bsz, 1, device=device, dtype=z_cam.dtype)
            mask_lidar = torch.ones(bsz, 1, device=device, dtype=z_lidar.dtype)
            return z_cam, z_lidar, torch.cat([mask_cam, mask_lidar], dim=-1)

        keep_cam = torch.bernoulli(torch.full((bsz, 1), 1 - self.p_cam, device=device, dtype=z_cam.dtype))
        keep_lidar = torch.bernoulli(torch.full((bsz, 1), 1 - self.p_lidar, device=device, dtype=z_lidar.dtype))

        both_zero = (keep_cam + keep_lidar) == 0
        if both_zero.any():
            random_choice = torch.bernoulli(torch.full((bsz, 1), 0.5, device=device, dtype=z_cam.dtype))
            keep_cam = torch.where(both_zero, random_choice, keep_cam)
            keep_lidar = torch.where(both_zero, 1.0 - random_choice, keep_lidar)

        z_cam = z_cam * keep_cam[:, :, None, None]
        z_lidar = z_lidar * keep_lidar[:, :, None, None]
        return z_cam, z_lidar, torch.cat([keep_cam, keep_lidar], dim=-1)


class AlignmentProxy(nn.Module):
    """Alignment quality proxy without GT metadata."""

    def __init__(self, mode: str = "none", proj_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.proj_cam = nn.LazyConv2d(proj_dim, kernel_size=1, bias=False)
        self.proj_lidar = nn.LazyConv2d(proj_dim, kernel_size=1, bias=False)

    @property
    def proxy_dim(self) -> int:
        return 3

    def forward(self, z_cam: torch.Tensor, z_lidar: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        run_mode = self.mode if mode is None else mode
        bsz = z_cam.shape[0]
        if run_mode == "none":
            return z_cam.new_zeros((bsz, self.proxy_dim))
        if run_mode == "geom_overlap":
            return z_cam.new_zeros((bsz, self.proxy_dim))
        if run_mode != "feat_cosine":
            raise ValueError(f"Unsupported alignment proxy mode: {run_mode}")

        p_cam = self.proj_cam(z_cam)
        p_lidar = self.proj_lidar(z_lidar)

        v_cam = p_cam.mean(dim=(-2, -1))
        v_lidar = p_lidar.mean(dim=(-2, -1))

        v_cam_n = F.normalize(v_cam, p=2, dim=-1, eps=self.eps)
        v_lidar_n = F.normalize(v_lidar, p=2, dim=-1, eps=self.eps)

        q_cos = (v_cam_n * v_lidar_n).sum(dim=-1, keepdim=True)
        q_ec = torch.log(self.eps + v_cam.norm(p=2, dim=-1, keepdim=True))
        q_el = torch.log(self.eps + v_lidar.norm(p=2, dim=-1, keepdim=True))

        return torch.cat([q_cos, q_ec, q_el], dim=-1)


class GlobalGatedFusion(nn.Module):
    """Global two-modality gate with optional mask/alignment proxy input."""

    def __init__(
        self,
        hidden_dim: int = 128,
        use_alignment_proxy: bool = False,
        alignment_proxy_mode: str = "none",
        gate_use_mask: bool = True,
        gate_proj_dim: int = 64,
    ):
        super().__init__()
        self.use_alignment_proxy = use_alignment_proxy
        self.gate_use_mask = gate_use_mask
        self.alignment_proxy = AlignmentProxy(mode=alignment_proxy_mode, proj_dim=gate_proj_dim)

        extra_dim = (2 if gate_use_mask else 0) + self.alignment_proxy.proxy_dim
        self.gate = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )
        self.extra_dim = extra_dim

    def forward(
        self,
        z_cam: torch.Tensor,
        z_lidar: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pool_cam = z_cam.mean(dim=(-2, -1))
        pool_lidar = z_lidar.mean(dim=(-2, -1))

        gate_inputs = [pool_cam, pool_lidar]

        if self.gate_use_mask:
            if modality_mask is None:
                modality_mask = z_cam.new_ones((z_cam.shape[0], 2))
            gate_inputs.append(modality_mask)

        if self.use_alignment_proxy:
            proxy = self.alignment_proxy(z_cam, z_lidar)
        else:
            proxy = z_cam.new_zeros((z_cam.shape[0], self.alignment_proxy.proxy_dim))
        gate_inputs.append(proxy)

        gate_in = torch.cat(gate_inputs, dim=-1)
        alpha = torch.softmax(self.gate(gate_in), dim=-1)

        z_fused = torch.cat([
            z_cam * alpha[:, 0].view(-1, 1, 1, 1),
            z_lidar * alpha[:, 1].view(-1, 1, 1, 1),
        ], dim=1)
        return z_fused, alpha


class SparseMoESpatialGate(nn.Module):
    """
    Sparse MoE Top-K hard spatial gate with STE for Camera-LiDAR BEV features.

    References:
      - Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
        https://arxiv.org/abs/1701.06538
      - Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
        https://arxiv.org/abs/2312.00752
      - MoE-Mamba: https://arxiv.org/abs/2401.04081
    """

    def __init__(self, hidden_dim: int = 64, topk: int = 1, include_null_expert: bool = True):
        super().__init__()
        self.topk = int(topk)
        self.include_null_expert = include_null_expert
        self.num_experts = 3 if include_null_expert else 2  # [cam, lidar, null]
        self.router = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.num_experts)
        )

    def _topk_hard_gate(self, probs: torch.Tensor) -> torch.Tensor:
        topk = max(1, min(self.topk, probs.shape[-1]))
        topk_idx = torch.topk(probs, k=topk, dim=-1).indices
        hard_mask = torch.zeros_like(probs).scatter_(-1, topk_idx, 1.0)
        sparse_probs = probs * hard_mask
        # STE: forward uses hard Top-K sparse probabilities; backward uses selected probability path.
        sparse_probs_ste = sparse_probs.detach() + (sparse_probs - sparse_probs.detach())
        return sparse_probs_ste

    def forward(
        self,
        z_cam: torch.Tensor,
        z_lidar: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, _, h, w = z_cam.shape
        x_cam = z_cam.permute(0, 2, 3, 1).reshape(bsz, h * w, -1)
        x_lidar = z_lidar.permute(0, 2, 3, 1).reshape(bsz, h * w, -1)

        token_feat = torch.cat([x_cam, x_lidar], dim=-1)
        logits = self.router(token_feat)

        if modality_mask is not None:
            modality_mask = modality_mask.to(dtype=logits.dtype)
            mask_logits = torch.zeros_like(logits)
            mask_logits[..., 0] = (modality_mask[:, 0:1] - 1.0) * 1e4
            mask_logits[..., 1] = (modality_mask[:, 1:2] - 1.0) * 1e4
            logits = logits + mask_logits

        probs = torch.softmax(logits, dim=-1)
        gate = self._topk_hard_gate(probs)

        gate_cam = gate[..., 0:1]
        gate_lidar = gate[..., 1:2]
        keep_mask = ((gate_cam + gate_lidar) > 0).to(dtype=z_cam.dtype)

        xhat_cam = x_cam * gate_cam
        xhat_lidar = x_lidar * gate_lidar

        zhat_cam = xhat_cam.reshape(bsz, h, w, -1).permute(0, 3, 1, 2).contiguous()
        zhat_lidar = xhat_lidar.reshape(bsz, h, w, -1).permute(0, 3, 1, 2).contiguous()
        keep_mask_2d = keep_mask.reshape(bsz, h, w, 1).permute(0, 3, 1, 2).contiguous()

        gate_stats = {
            'router_prob': probs,
            'router_gate': gate,
            'keep_ratio': keep_mask.float().mean(dim=1),
        }
        return zhat_cam, zhat_lidar, keep_mask_2d, gate_stats
