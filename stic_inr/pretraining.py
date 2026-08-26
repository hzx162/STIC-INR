"""Offline meta-pretraining for the initialization used by online adaptation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .adaptation import online_adapt


@dataclass(frozen=True)
class SnapshotTask:
    """One CFD time step treated as a meta-learning task."""

    coordinates: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class PretrainingResult:
    model: nn.Module
    outer_losses: tuple[float, ...]


def meta_pretrain(
    model: nn.Module,
    tasks: Sequence[SnapshotTask],
    *,
    outer_steps: int = 200,
    outer_learning_rate: float = 1e-4,
    inner_learning_rate: float = 1e-4,
    inner_steps: int = 30,
    batch_size: int | None = 50_000,
    tasks_per_outer_step: int | None = None,
    seed: int = 42,
) -> PretrainingResult:
    """Learn an INR initialization using a first-order Reptile update.

    Every snapshot is a separate task. For each outer iteration, task learners
    are copied from the shared initialization and adapted independently. The
    shared model is then moved toward the mean of the adapted task parameters.
    The returned model can be used directly as the initialization for
    ``online_adapt``.
    """
    if not tasks:
        raise ValueError("at least one snapshot task is required")
    if outer_steps < 1:
        raise ValueError("outer_steps must be positive")
    if tasks_per_outer_step is not None and tasks_per_outer_step < 1:
        raise ValueError("tasks_per_outer_step must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=outer_learning_rate)
    loss_history: list[float] = []

    for _ in range(outer_steps):
        if tasks_per_outer_step is None or tasks_per_outer_step >= len(tasks):
            selected = list(tasks)
        else:
            indices = torch.randperm(len(tasks), generator=generator)[
                :tasks_per_outer_step
            ].tolist()
            selected = [tasks[index] for index in indices]

        gradient_sums = [torch.zeros_like(parameter) for parameter in model.parameters()]
        task_losses: list[float] = []

        for task in selected:
            learner = deepcopy(model)
            adapted = online_adapt(
                learner,
                task.coordinates,
                task.values,
                learning_rate=inner_learning_rate,
                adaptation_steps=inner_steps,
                batch_size=batch_size,
                in_place=True,
            )
            task_losses.append(adapted.mse)
            with torch.no_grad():
                for gradient, meta_parameter, task_parameter in zip(
                    gradient_sums,
                    model.parameters(),
                    adapted.model.parameters(),
                    strict=True,
                ):
                    # Gradient descent on (meta - adapted) moves the shared
                    # initialization toward the task-adapted parameters.
                    gradient.add_(meta_parameter - task_parameter)

        outer_optimizer.zero_grad(set_to_none=True)
        scale = 1.0 / len(selected)
        for parameter, gradient in zip(
            model.parameters(), gradient_sums, strict=True
        ):
            parameter.grad = gradient.mul(scale)
        outer_optimizer.step()
        loss_history.append(sum(task_losses) / len(task_losses))

    return PretrainingResult(model=model, outer_losses=tuple(loss_history))

