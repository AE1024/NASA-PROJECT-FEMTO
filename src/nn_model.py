"""
Weibull-hybrid feedforward NN for RUL prediction.
Reference: von Hahn & Mechefske (2022) arXiv:2201.01769
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid":    nn.Sigmoid,
    "silu":       nn.SiLU,
}


class NN_Block(nn.Module):
    """Linear → [activation] → Dropout."""

    def __init__(self, input_dim: int, output_dim: int, activation: nn.Module | None, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        return x


class RULnet(nn.Module):
    """
    Feedforward network: input → n_layers hidden blocks → scalar RUL_pct ∈ [0, 100].

    Architecture (Makale 1):
      block_0 : Linear(input_dim → n_units) + activation + dropout   ← first hidden
      block_1…n_layers-1 : Linear(n_units → n_units) + activation + dropout
      output  : Linear(n_units → 1) → sigmoid × 100

    Parameters
    ----------
    input_dim    : number of input features (default 21)
    n_layers     : total hidden blocks (≥ 1)
    n_units      : hidden units per block
    dropout_rate : dropout probability [0, 1)
    activation   : str (same for all blocks) or list[str] of length n_layers
    """

    def __init__(
        self,
        input_dim: int = 21,
        n_layers: int = 3,
        n_units: int = 64,
        dropout_rate: float = 0.1,
        activation: str | list[str] = "relu",
    ):
        super().__init__()
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1): {dropout_rate}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1: {n_layers}")

        # Normalize activation to list[str] of length n_layers
        if isinstance(activation, str):
            activations: list[str] = [activation] * n_layers
        else:
            if len(activation) != n_layers:
                raise ValueError(
                    f"activation list length ({len(activation)}) must equal n_layers ({n_layers})"
                )
            activations = list(activation)

        for a in activations:
            if a not in _ACTIVATIONS:
                raise ValueError(f"Unknown activation '{a}'. Choose from {list(_ACTIVATIONS)}")

        self.blocks = nn.ModuleList()
        # First block: input_dim → n_units
        self.blocks.append(
            NN_Block(input_dim, n_units, _ACTIVATIONS[activations[0]](), dropout_rate)
        )
        # Remaining hidden blocks: n_units → n_units
        for act in activations[1:]:
            self.blocks.append(
                NN_Block(n_units, n_units, _ACTIVATIONS[act](), dropout_rate)
            )

        self.output_layer = nn.Linear(n_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        # sigmoid → scale to [0, 100] (LifePercentage)
        return torch.sigmoid(self.output_layer(x)).squeeze(-1) * 100.0


class WeibullLoss(nn.Module):
    """
    Hybrid loss: MSE + λ · mean|F(pred) − F(true)|

    Weibull CDF: F(t) = 1 − exp(−(t / η)^β)

    Units: predictions and targets are LifePercentage ∈ [0, 100].
    β and η are fit on train bearing lifetimes expressed as LifePercentage,
    so η ≈ 50.86 (not the raw step count).

    Reference: eq. (4–5) in arXiv:2201.01769
    """

    # Fit from train lifetimes [516, 798, 872, 912, 1638, 2803] steps,
    # converted to LifePercentage (÷ 2803 × 100):
    #   → weibull_min.fit([18.41, 28.47, 31.11, 32.54, 58.44, 100.0], floc=0)
    _BETA_DEFAULT: float = 1.7801
    _ETA_DEFAULT: float = 50.8620   # LifePercentage units

    def __init__(
        self,
        lambda_w: float,
        beta: float = _BETA_DEFAULT,
        eta: float = _ETA_DEFAULT,
    ):
        super().__init__()
        if lambda_w < 0:
            raise ValueError(f"lambda_w must be ≥ 0: {lambda_w}")
        self.lambda_w = lambda_w
        self.beta = beta
        self.eta = eta

    def _weibull_cdf(self, t: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid log(0) when t=0
        t_safe = torch.clamp(t, min=1e-6)
        return 1.0 - torch.exp(-((t_safe / self.eta) ** self.beta))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(predictions, targets)
        if self.lambda_w == 0.0:
            return mse
        F_pred = self._weibull_cdf(predictions)
        F_true = self._weibull_cdf(targets)
        weibull_term = torch.mean(torch.abs(F_pred - F_true))
        return mse + self.lambda_w * weibull_term
