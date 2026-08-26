"""Offline Meta-INR pretraining from a sequence stored as NumPy arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from stic_inr import SIREN, SnapshotTask, meta_pretrain
from stic_inr.adaptation import normalize_coordinates, normalize_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline STIC-INR meta-pretraining")
    parser.add_argument("--coordinates", type=Path, required=True, help="NPY array [N, d]")
    parser.add_argument(
        "--snapshots",
        type=Path,
        required=True,
        help="NPY array [T, N] for a scalar field or [T, N, c] for a vector field",
    )
    parser.add_argument("--output", type=Path, default=Path("initial_model.pth"))
    parser.add_argument("--outer-steps", type=int, default=200)
    parser.add_argument("--inner-steps", type=int, default=30)
    parser.add_argument("--outer-lr", type=float, default=1e-4)
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--tasks-per-outer-step", type=int)
    parser.add_argument("--init-features", type=int, default=8)
    parser.add_argument("--hidden-layers", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coordinates_np = np.load(args.coordinates)
    snapshots_np = np.load(args.snapshots)
    if coordinates_np.ndim != 2:
        raise ValueError("coordinates must have shape [N, d]")
    if snapshots_np.ndim not in (2, 3):
        raise ValueError("snapshots must have shape [T, N] or [T, N, c]")
    if snapshots_np.shape[1] != coordinates_np.shape[0]:
        raise ValueError("coordinates and snapshots contain different point counts")

    coordinates = normalize_coordinates(torch.from_numpy(coordinates_np).float()).to(device)
    tasks = [
        SnapshotTask(
            coordinates=coordinates,
            values=normalize_field(torch.from_numpy(snapshot).float()).to(device),
        )
        for snapshot in snapshots_np
    ]
    output_features = 1 if snapshots_np.ndim == 2 else snapshots_np.shape[2]
    model = SIREN(
        in_features=coordinates_np.shape[1],
        out_features=output_features,
        init_features=args.init_features,
        num_hidden_layers=args.hidden_layers,
    ).to(device)

    result = meta_pretrain(
        model,
        tasks,
        outer_steps=args.outer_steps,
        outer_learning_rate=args.outer_lr,
        inner_learning_rate=args.inner_lr,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        tasks_per_outer_step=args.tasks_per_outer_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), args.output)
    print(f"saved initialization: {args.output}")
    print(f"final mean task MSE: {result.outer_losses[-1]:.6e}")


if __name__ == "__main__":
    main()

