"""Coordinate-based implicit neural representation used by STIC-INR."""

from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    """Linear layer followed by a sinusoidal activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(coordinates))


class OutputLinear(nn.Module):
    """Final linear mapping with the initialization used in the experiments."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = 1.0 / in_features
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class SIREN(nn.Module):
    """SIREN mapping spatial coordinates to one CFD field snapshot.

    Parameters follow the architecture used by the method: the hidden width is
    ``4 * init_features`` and ``num_hidden_layers`` denotes the number of
    non-first sine layers.
    """

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        init_features: int = 8,
        num_hidden_layers: int = 5,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        width = 4 * init_features
        layers: list[nn.Module] = [
            SineLayer(
                in_features,
                width,
                is_first=True,
                omega_0=omega_0,
            )
        ]
        layers.extend(
            SineLayer(width, width, omega_0=omega_0)
            for _ in range(num_hidden_layers)
        )
        layers.append(OutputLinear(width, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.net(coordinates)
