#arxiv.org/abs/2201.01769 article deep learning solution implementation
import torch
import torch.nn as nn
_ACTIVATIONS = {
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid" : nn.Sigmoid,
    "SiLU": nn.SiLU
}

class NN_Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation, dropout_rate):
        super().__init__()
        self.linear= nn.Linear(input_dim, output_dim)
        self.activation = activation if activation else None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        return x

class RULnet(nn.Module):
    def __init__(self, input_dim: int = 21, n_layers: int = 3, n_units: int = 32, dropout_rate: float = 0.1, activation: str = "relu"):
        super().__init__()
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate should be 0-1 : {dropout_rate}")
        if isinstance(activation, str):
            activations = [activation] * (n_layers - 1)
        else:
            if len(activation) != n_layers - 1:
                raise ValueError(f"activation list should be n_layers-1={n_layers-1} .")
            activations = activation


        self.blocks = nn.ModuleList()
        # Input block
        self.blocks.append(NN_Block(input_dim, n_units, activation=None, dropout_rate=dropout_rate))

        # Hidden blocks
        for act in activations:
            self.blocks.append(NN_Block(n_units, n_units, activation=_ACTIVATIONS[act](), dropout_rate=dropout_rate))
        # Output layer
        self.output_layer = nn.Linear(n_units, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = torch.sigmoid(self.output_layer(x))
        return x * 100

#model = RULnet(input_dim=21, n_layers=5, n_units=32, dropout_rate=0.2, activation=["relu","tanh","relu","relu"])
#print(model)
#x = torch.randn(8, 21)   # batch=8, feature=21
#print(model(x).shape)    # → torch.Size([8, 1])

