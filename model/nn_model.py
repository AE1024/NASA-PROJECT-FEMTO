# arxiv.org/abs/2201.01769 — feedforward NN for RUL prediction
import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu":  nn.ReLU,
    "tanh":  nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "silu":  nn.SiLU,
}


class NN_Block(nn.Module):
    """Single hidden block: Linear -> Activation -> Dropout."""

    def __init__(self, input_dim: int, output_dim: int, activation: nn.Module, dropout_rate: float):
        super().__init__()
        self.linear     = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class RULnet(nn.Module):
    """
    Feedforward network for RUL prediction (von Hahn & Mechefske, 2022).

    Architecture:
        input (input_dim)
        -> n_layers x [Linear -> Activation -> Dropout]
        -> Linear(n_units, 1) -> Sigmoid x 100
        -> output: LifePercentage in [0, 100]

    Parameters
    ----------
    input_dim    : number of input features
    n_layers     : number of hidden blocks (>= 1)
    n_units      : hidden units per block
    dropout_rate : dropout probability, in [0, 1)
    activation   : activation name (applied to all blocks) or list of names
                   (one per block, length must equal n_layers).
                   Choices: relu, tanh, leaky_relu, sigmoid, silu
    """

    def __init__(
        self,
        input_dim: int= 21,
        n_layers: int  = 3,
        n_units: int  = 64,
        dropout_rate: float = 0.1,
        activation: str | list[str] = "relu",
    ):
        super().__init__()
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1): {dropout_rate}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1: {n_layers}")

        if isinstance(activation, str):
            activations: list[str] = [activation] * n_layers
        else:
            if len(activation) != n_layers:
                raise ValueError(
                    f"activation list length must equal n_layers={n_layers}, got {len(activation)}"
                )
            activations = list(activation)

        for a in activations:
            if a not in _ACTIVATIONS:
                raise ValueError(f"Unknown activation '{a}'. Choices: {list(_ACTIVATIONS)}")

        self.blocks = nn.ModuleList()
        self.blocks.append(NN_Block(input_dim, n_units, _ACTIVATIONS[activations[0]](), dropout_rate))
        for act in activations[1:]:
            self.blocks.append(NN_Block(n_units, n_units, _ACTIVATIONS[act](), dropout_rate))

        self.output_layer = nn.Linear(n_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.output_layer(x)).squeeze(-1) * 100
