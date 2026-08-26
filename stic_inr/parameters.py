"""Stable conversion between an INR and its high-dimensional parameter vector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class ParameterLayout:
    """Parameter order and shapes required to reconstruct a model."""

    parameters: tuple[ParameterSpec, ...]

    @property
    def size(self) -> int:
        return sum(item.size for item in self.parameters)


def flatten_parameters(model: nn.Module) -> tuple[np.ndarray, ParameterLayout]:
    """Flatten trainable parameters in deterministic ``named_parameters`` order."""
    arrays: list[np.ndarray] = []
    specs: list[ParameterSpec] = []
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().reshape(-1).double().numpy()
        arrays.append(value)
        specs.append(ParameterSpec(name, tuple(parameter.shape), parameter.numel()))
    if not arrays:
        raise ValueError("model contains no trainable parameters")
    return np.concatenate(arrays), ParameterLayout(tuple(specs))


def restore_parameters(
    model: nn.Module,
    vector: np.ndarray,
    layout: ParameterLayout,
) -> nn.Module:
    """Restore ``vector`` into a model with the same parameter structure."""
    vector = np.asarray(vector).reshape(-1)
    if vector.size != layout.size:
        raise ValueError(f"parameter size mismatch: expected {layout.size}, got {vector.size}")

    current = dict(model.named_parameters())
    offset = 0
    with torch.no_grad():
        for spec in layout.parameters:
            if spec.name not in current:
                raise ValueError(f"model is missing parameter {spec.name!r}")
            parameter = current[spec.name]
            if tuple(parameter.shape) != spec.shape:
                raise ValueError(f"shape mismatch for parameter {spec.name!r}")
            chunk = vector[offset : offset + spec.size].reshape(spec.shape)
            parameter.copy_(torch.as_tensor(chunk, device=parameter.device, dtype=parameter.dtype))
            offset += spec.size
    return model

