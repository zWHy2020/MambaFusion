import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
