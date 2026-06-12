"""
Feature selector modules for Stage 2 ablation study.

All four selectors share the same interface:
  forward(x)       → gated features, same shape as x [B, D]
  get_reg_loss()   → scalar regularisation loss (0 for SE / ECA)

Selectors
---------
SEGate            SE-Net style FC bottleneck gate        74,544 params (D=768, r=16)
L0HardConcreteGate  L0 Hard Concrete per-feature gates      768 params
ECAGate           ECA-Net adaptive 1D-conv gate              5 params  (k=9 for D=768)
ECASTGHybrid      ECA → Stochastic Gates pipeline          773 params
"""

import math
import torch
import torch.nn as nn


# =============================================================================
# SE Gate
# =============================================================================

class SEGate(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    FC(D→D/r) → ReLU → Dropout → FC(D/r→D) → Sigmoid
    """

    def __init__(self, in_features: int = 768, reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        hidden = max(in_features // reduction, 32)
        self.gate = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_features),
            nn.Sigmoid(),
        )
        # Initialise: sigmoid starts near 0.5 → approximate pass-through
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.gate[-2].bias, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)

    def get_reg_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


# =============================================================================
# L0 Hard Concrete Gate
# =============================================================================

class L0HardConcreteGate(nn.Module):
    """
    L0 Hard Concrete per-feature stochastic gates (Louizos et al., ICLR 2018).
    768 learnable log_alpha parameters.
    Closed-form L0 regularisation penalty encourages sparsity.
    """

    def __init__(
        self,
        in_features: int = 768,
        reg_lambda: float = 0.01,
        beta: float = 2.0 / 3.0,
        zeta: float = 1.1,
        gamma: float = -0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.reg_lambda  = reg_lambda
        self.beta        = beta
        self.zeta        = zeta
        self.gamma       = gamma
        self.log_alpha   = nn.Parameter(torch.full((in_features,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta
            )
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            z = torch.clamp(s_bar, 0.0, 1.0)
        else:
            z = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
            z = torch.clamp(z, 0.0, 1.0)
        return x * z.unsqueeze(0)

    def get_reg_loss(self) -> torch.Tensor:
        return self.reg_lambda * torch.sum(
            torch.sigmoid(
                self.log_alpha - self.beta * math.log(-self.gamma / self.zeta)
            )
        )

    def get_gate_stats(self) -> dict:
        with torch.no_grad():
            z = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
            z = torch.clamp(z, 0.0, 1.0)
            active = (z > 0.01).sum().item()
            dead   = (z < 0.01).sum().item()
            return {"active": active, "dead": dead,
                    "sparsity": dead / self.in_features}


# =============================================================================
# ECA Gate
# =============================================================================

class ECAGate(nn.Module):
    """
    ECA-Net adaptive 1D-conv channel attention (Wang et al., CVPR 2020).
    Kernel size k derived from feature dimension D:
      k = 9 for D = 768.
    Only 5 trainable parameters regardless of D.
    """

    def __init__(self, in_features: int = 768, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs(math.log2(in_features) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        self.kernel_size = k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        nn.init.ones_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.conv(x.unsqueeze(1)).squeeze(1))
        return x * gate

    def get_reg_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


# =============================================================================
# Stochastic Gates
# =============================================================================

class StochasticGates(nn.Module):
    """
    STG — per-feature stochastic gates with Gaussian noise (Yamada et al., ICML 2020).
    768 learnable mu parameters.
    """

    def __init__(
        self,
        in_features: int = 768,
        sigma: float = 0.5,
        reg_lambda: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.sigma       = sigma
        self.reg_lambda  = reg_lambda
        self.mu          = nn.Parameter(torch.full((in_features,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(self.mu) * self.sigma
            z   = torch.clamp(self.mu + 0.5 + eps, 0.0, 1.0)
        else:
            z = torch.clamp(self.mu + 0.5, 0.0, 1.0)
        return x * z.unsqueeze(0)

    def get_reg_loss(self) -> torch.Tensor:
        return self.reg_lambda * torch.sum(
            0.5 * (1 + torch.erf(
                (self.mu + 0.5) / (self.sigma * math.sqrt(2))
            ))
        )

    def get_gate_stats(self) -> dict:
        with torch.no_grad():
            z = torch.clamp(self.mu + 0.5, 0.0, 1.0)
            active = (z > 0.01).sum().item()
            dead   = (z < 0.01).sum().item()
            return {"active": active, "dead": dead,
                    "sparsity": dead / self.in_features}


# =============================================================================
# ECA + STG Hybrid
# =============================================================================

class ECASTGHybrid(nn.Module):
    """
    Two-stage pipeline: ECA gate → Stochastic gate.
    Total params: 5 (ECA) + 768 (STG) = 773.
    """

    def __init__(
        self,
        in_features: int = 768,
        eca_gamma: int = 2,
        eca_b: int = 1,
        stg_sigma: float = 0.5,
        stg_lambda: float = 0.01,
    ):
        super().__init__()
        self.eca = ECAGate(in_features, eca_gamma, eca_b)
        self.stg = StochasticGates(in_features, stg_sigma, stg_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stg(self.eca(x))

    def get_reg_loss(self) -> torch.Tensor:
        return self.stg.get_reg_loss()


# =============================================================================
# Factory
# =============================================================================

SELECTOR_REGISTRY = {
    "se_gate":  "SEGate",
    "l0":       "L0HardConcreteGate",
    "eca":      "ECAGate",
    "eca_stg":  "ECASTGHybrid",
    "none":     None,           # baseline: no selector
}


def create_selector(name: str, in_features: int = 768) -> nn.Module:
    """
    Instantiate a selector by name.
    name='none' returns an identity module (used for baseline experiment).
    """
    if name == "se_gate":
        return SEGate(in_features, reduction=16, dropout=0.1)
    elif name == "l0":
        return L0HardConcreteGate(in_features, reg_lambda=0.01)
    elif name == "eca":
        return ECAGate(in_features, gamma=2, b=1)
    elif name == "eca_stg":
        return ECASTGHybrid(in_features, stg_sigma=0.5, stg_lambda=0.01)
    elif name == "none":
        return nn.Identity()
    else:
        raise ValueError(
            f"Unknown selector '{name}'. "
            f"Choose from: {list(SELECTOR_REGISTRY.keys())}"
        )


def selector_param_count(name: str, in_features: int = 768) -> int:
    """Return the parameter count for a given selector."""
    sel = create_selector(name, in_features)
    return sum(p.numel() for p in sel.parameters())
