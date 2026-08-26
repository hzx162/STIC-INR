"""Reference orchestration for OpenFOAM -> SmartSim -> STIC-INR.

This file documents the integration contract used by the method. It is not a
turnkey case runner: users must provide a compiled ``fieldsToSmartRedis``
OpenFOAM function object, a case-specific checkpoint, and exported mesh
coordinates. Dataset naming is obtained from the function object's metadata
rather than being hard-coded.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from stic_inr import (
    SIREN,
    StreamingWeightTrajectoryCompressor,
    flatten_parameters,
    online_adapt,
)
from stic_inr.adaptation import normalize_coordinates, normalize_field


@dataclass(frozen=True)
class RuntimeConfig:
    # OpenFOAM/SmartSim integration supplied by the user.
    case_dir: Path
    foam_bashrc: Path
    solver: str
    function_object: str
    field: str
    patch: str = "internal"

    # Method inputs. Coordinates are deliberately accepted as a portable NPY
    # file so the public method does not depend on a private OpenFOAM parser.
    coordinates_file: Path = Path("mesh_coordinates.npy")
    initial_checkpoint: Path = Path("initial_model.pth")
    coordinate_dimension: int = 3
    field_components: int = 1
    init_features: int = 8
    hidden_layers: int = 5

    # Runtime and method configuration.
    db_port: int = 6780
    db_interface: str = "lo"
    first_step: int = 0
    step_stride: int = 1
    max_snapshots: int = 100
    poll_ms: int = 500
    learning_rate: float = 1e-4
    adaptation_steps: int = 20
    batch_size: int = 50_000
    compression_threshold: float | None = None
    compression_auto_multiplier: float = 2.0
    delete_consumed_snapshots: bool = True


@dataclass(frozen=True)
class Snapshot:
    step: int
    values: np.ndarray
    dataset_name: str


class SmartSimOpenFOAMStream:
    """Small adapter exposing OpenFOAM output as a Python snapshot iterator."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.experiment: Any = None
        self.database: Any = None
        self.foam_model: Any = None
        self.client: Any = None
        self._render_names: Any = None

    @staticmethod
    def _runtime_types() -> tuple[Any, Any, Any, Any, Any]:
        try:
            from jinja2 import Template
            from smartsim import Experiment
            from smartsim.database import Orchestrator
            from smartsim.settings import RunSettings
            from smartredis import Client
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional packages in requirements-smartsim.txt"
            ) from exc
        return Experiment, Orchestrator, RunSettings, Client, Template

    def start(self) -> None:
        """Start the database, expose SSDB, and launch the OpenFOAM solver."""
        Experiment, Orchestrator, RunSettings, Client, _ = self._runtime_types()
        cfg = self.config

        self.experiment = Experiment("stic_inr_reference", launcher="local")
        self.database = Orchestrator(port=cfg.db_port, interface=cfg.db_interface)
        self.experiment.start(self.database)
        database_address = self.database.get_address()[0]

        # fieldsToSmartRedis reads SSDB from the solver environment. Quoting is
        # kept here because paths are integration inputs, not method constants.
        command = " && ".join(
            [
                f"source {shlex.quote(str(cfg.foam_bashrc))}",
                f"export SSDB={shlex.quote(database_address)}",
                f"{shlex.quote(cfg.solver)} -case {shlex.quote(str(cfg.case_dir))}",
            ]
        )
        settings = RunSettings("bash", exe_args=["-lc", command])
        self.foam_model = self.experiment.create_model(
            "openfoam",
            settings,
            path=str(cfg.case_dir.parent),
        )
        self.experiment.start(self.foam_model, block=False)
        self.client = Client(address=database_address, cluster=False)
        self._render_names = self._read_name_contract()

    def _read_name_contract(self) -> Any:
        """Read dataset/field templates published by fieldsToSmartRedis."""
        _, _, _, _, Template = self._runtime_types()
        metadata_name = f"{self.config.function_object}_metadata"
        while not self.client.poll_dataset(metadata_name, self.config.poll_ms, 1):
            if self.solver_finished:
                raise RuntimeError("OpenFOAM stopped before publishing metadata")

        metadata = self.client.get_dataset(metadata_name)
        dataset_template = Template(metadata.get_meta_strings("dataset")[0])
        field_template = Template(metadata.get_meta_strings("field")[0])

        def render(step: int) -> tuple[str, str]:
            dataset_name = dataset_template.render(time_index=step, mpi_rank=0)
            tensor_name = field_template.render(
                name=self.config.field,
                patch=self.config.patch,
            )
            return dataset_name, tensor_name

        return render

    @property
    def solver_finished(self) -> bool:
        if self.foam_model is None:
            return False
        return self.experiment.get_status(self.foam_model)[0] == "Completed"

    def snapshots(self) -> Iterator[Snapshot]:
        """Yield at most ``max_snapshots`` snapshots using the metadata contract."""
        step = self.config.first_step
        emitted = 0
        while emitted < self.config.max_snapshots:
            dataset_name, tensor_name = self._render_names(step)
            if not self.client.poll_dataset(dataset_name, self.config.poll_ms, 1):
                if self.solver_finished:
                    return
                continue

            dataset = self.client.get_dataset(dataset_name)
            if tensor_name not in dataset.get_tensor_names():
                raise RuntimeError(
                    f"{dataset_name!r} does not contain expected tensor {tensor_name!r}"
                )
            yield Snapshot(step, np.asarray(dataset.get_tensor(tensor_name)), dataset_name)

            if self.config.delete_consumed_snapshots:
                self.client.delete_dataset(dataset_name)
            emitted += 1
            step += self.config.step_stride

    def publish_compression(self, update: Any) -> None:
        """Publish the current compressed model-weight trajectory."""
        from smartredis import Dataset

        segments = np.asarray(
            [(segment.start, segment.end) for segment in update.segments],
            dtype=np.int64,
        ).reshape(-1, 2)
        dataset = Dataset(f"stic_inr_weight_trajectory_{update.step}")
        dataset.add_tensor("segments", segments)
        dataset.add_tensor(
            "segment_starts",
            np.asarray(update.segment_starts, dtype=np.int64),
        )
        dataset.add_tensor(
            "segment_slopes",
            np.stack([segment.slope for segment in update.segments]),
        )
        dataset.add_tensor(
            "segment_intercepts",
            np.stack([segment.intercept for segment in update.segments]),
        )
        dataset.add_meta_scalar(
            "starts_new_segment",
            int(update.starts_new_segment),
        )
        dataset.add_meta_scalar(
            "fitting_residual",
            float(update.latest_residual),
        )
        self.client.put_dataset(dataset)

    def stop(self) -> None:
        """Release only the runtime resources created by this adapter."""
        if self.experiment is None:
            return
        if self.foam_model is not None:
            self.experiment.stop(self.foam_model)
        if self.database is not None:
            self.experiment.stop(self.database)


def load_initial_model(config: RuntimeConfig, device: torch.device) -> SIREN:
    """Case-specific model construction kept outside the transport adapter."""
    model = SIREN(
        in_features=config.coordinate_dimension,
        out_features=config.field_components,
        init_features=config.init_features,
        num_hidden_layers=config.hidden_layers,
    )
    state = torch.load(config.initial_checkpoint, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def run_online_compression(config: RuntimeConfig) -> None:
    """Compose transport, online adaptation, and weight-trajectory compression."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordinates_np = np.load(config.coordinates_file)
    coordinates = normalize_coordinates(torch.from_numpy(coordinates_np).float()).to(device)
    model = load_initial_model(config, device)
    compressor = StreamingWeightTrajectoryCompressor(
        config.compression_threshold,
        auto_multiplier=config.compression_auto_multiplier,
    )
    stream = SmartSimOpenFOAMStream(config)

    stream.start()
    try:
        for snapshot in stream.snapshots():
            values = normalize_field(torch.from_numpy(snapshot.values).float()).to(device)
            adapted = online_adapt(
                model,
                coordinates,
                values,
                learning_rate=config.learning_rate,
                adaptation_steps=config.adaptation_steps,
                batch_size=config.batch_size,
                in_place=True,
            )
            model = adapted.model
            parameter_vector, _ = flatten_parameters(model)
            update = compressor.update(snapshot.step, parameter_vector)
            stream.publish_compression(update)
            print(
                f"step={snapshot.step} mse={adapted.mse:.6e} "
                f"new_segment={update.starts_new_segment} "
                f"segments={len(update.segments)}"
            )
    finally:
        stream.stop()


if __name__ == "__main__":
    # These values describe the required integration inputs. They are
    # intentionally placeholders rather than paths from an internal system.
    run_online_compression(
        RuntimeConfig(
            case_dir=Path("/path/to/openfoam/case"),
            foam_bashrc=Path("/path/to/OpenFOAM/etc/bashrc"),
            solver="pisoFoam",
            function_object="pUPhiTest",
            field="U",
            coordinates_file=Path("/path/to/mesh_coordinates.npy"),
            initial_checkpoint=Path("/path/to/initial_model.pth"),
            field_components=3,
        )
    )
