"""Online adaptation of an INR to a newly arrived field snapshot."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AdaptationResult:
    """Result returned after adapting to one time step."""

    model: nn.Module
    loss: float
    mse: float
    updates: int


def normalize_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """Normalize every coordinate axis independently to ``[-1, 1]``."""
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape [num_points, dimensions]")
    lower = coordinates.amin(dim=0, keepdim=True)
    upper = coordinates.amax(dim=0, keepdim=True)
    span = upper - lower
    safe_span = torch.where(span.abs() < 1e-12, torch.ones_like(span), span)
    return 2.0 * (coordinates - lower) / safe_span - 1.0


def normalize_field(values: torch.Tensor) -> torch.Tensor:
    """Normalize a scalar or vector field snapshot globally to ``[-1, 1]``."""
    lower = values.amin()
    upper = values.amax()
    span = upper - lower
    if span.abs() < 1e-12:
        return torch.zeros_like(values)
    return 2.0 * (values - lower) / span - 1.0


def _aligned_values(values: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1 and prediction.ndim == 2 and prediction.shape[1] == 1:
        return values.unsqueeze(1)
    return values


def online_adapt(
    model: nn.Module,
    coordinates: torch.Tensor,
    values: torch.Tensor,
    *,
    learning_rate: float = 1e-4,
    adaptation_steps: int = 20,
    batch_size: int | None = 50_000,
    in_place: bool = True,
) -> AdaptationResult:
    """Adapt ``model`` to one normalized CFD snapshot.

    The function is independent of the in-situ transport layer. Coordinates and
    values may come from SmartSim, ADIOS2, Catalyst, or a file replay. Both are
    expected to already use the same normalization convention as offline
    training. Set ``in_place=True`` to inherit the previous time-step model.
    """
    if adaptation_steps < 1:
        raise ValueError("adaptation_steps must be positive")
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape [num_points, dimensions]")
    if values.shape[0] != coordinates.shape[0]:
        raise ValueError("coordinates and values must contain the same number of points")

    learner = model if in_place else deepcopy(model)
    device = next(learner.parameters()).device
    coordinates = coordinates.to(device=device, dtype=torch.float32)
    values = values.to(device=device, dtype=torch.float32)
    point_count = coordinates.shape[0]
    if point_count == 0:
        raise ValueError("a snapshot must contain at least one point")
    effective_batch_size = point_count if batch_size is None else min(batch_size, point_count)
    if effective_batch_size < 1:
        raise ValueError("batch_size must be positive")

    learner.train()
    optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    loss_sum = 0.0
    updates = 0

    for _ in range(adaptation_steps):
        permutation = torch.randperm(point_count, device=device)
        for start in range(0, point_count, effective_batch_size):
            indices = permutation[start : start + effective_batch_size]
            prediction = learner(coordinates[indices])
            target = _aligned_values(values[indices], prediction)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"model output shape {tuple(prediction.shape)} does not match "
                    f"field shape {tuple(target.shape)}"
                )
            loss = loss_function(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            updates += 1

    learner.eval()
    with torch.no_grad():
        prediction = learner(coordinates)
        target = _aligned_values(values, prediction)
        mse = float(loss_function(prediction, target))

    return AdaptationResult(
        model=learner,
        loss=loss_sum / updates,
        mse=mse,
        updates=updates,
    )
