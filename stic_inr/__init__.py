"""Core, infrastructure-independent components of STIC-INR."""

from .adaptation import AdaptationResult, online_adapt
from .model import SIREN
from .parameters import ParameterLayout, flatten_parameters, restore_parameters
from .pretraining import PretrainingResult, SnapshotTask, meta_pretrain
from .weight_trajectory import (
    CompressionUpdate,
    StreamingWeightTrajectoryCompressor,
    WeightTrajectorySegment,
)

__all__ = [
    "AdaptationResult",
    "online_adapt",
    "SIREN",
    "ParameterLayout",
    "flatten_parameters",
    "restore_parameters",
    "PretrainingResult",
    "SnapshotTask",
    "meta_pretrain",
    "CompressionUpdate",
    "StreamingWeightTrajectoryCompressor",
    "WeightTrajectorySegment",
]
