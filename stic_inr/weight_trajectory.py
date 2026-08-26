"""Streaming piecewise compression of the adapted model-weight trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class _StreamingLinearFitter:
    """Fit ``W(t) = slope * t + intercept`` using O(D) streaming updates."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.reset()

    def reset(self) -> None:
        self.ata = np.zeros((2, 2), dtype=np.float64)
        self.aty = np.zeros((2, self.dimension), dtype=np.float64)
        self.trace_yty = 0.0
        self.count = 0

    def copy(self) -> "_StreamingLinearFitter":
        clone = _StreamingLinearFitter(self.dimension)
        clone.ata = self.ata.copy()
        clone.aty = self.aty.copy()
        clone.trace_yty = self.trace_yty
        clone.count = self.count
        return clone

    def add(self, time: float, weight_vector: np.ndarray) -> None:
        weight_vector = np.asarray(weight_vector, dtype=np.float64).reshape(-1)
        if weight_vector.size != self.dimension:
            raise ValueError(
                f"expected dimension {self.dimension}, got {weight_vector.size}"
            )
        self.aty[0] += time * weight_vector
        self.aty[1] += weight_vector
        self.ata += np.array([[time * time, time], [time, 1.0]])
        self.trace_yty += float(np.dot(weight_vector, weight_vector))
        self.count += 1

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("cannot fit an empty trajectory segment")
        if self.count == 1:
            return np.zeros(self.dimension), self.aty[1].copy()
        slope, intercept = np.linalg.pinv(self.ata) @ self.aty
        return slope, intercept

    def squared_residual(self) -> float:
        if self.count < 3:
            return 0.0
        coefficients = np.linalg.pinv(self.ata) @ self.aty
        explained = float(np.sum(self.aty * coefficients))
        return max(0.0, self.trace_yty - explained)


@dataclass(frozen=True)
class WeightTrajectorySegment:
    """One compressed linear segment of the model-weight trajectory."""

    start: int
    end: int
    slope: np.ndarray
    intercept: np.ndarray

    def reconstruct(self, step: int) -> np.ndarray:
        if not self.start <= step <= self.end:
            raise ValueError(f"step {step} is outside [{self.start}, {self.end}]")
        return self.slope * (step - self.start) + self.intercept


@dataclass(frozen=True)
class CompressionUpdate:
    """Current state of streaming model-weight trajectory compression."""

    step: int
    starts_new_segment: bool
    latest_residual: float
    threshold: float | None
    segments: tuple[WeightTrajectorySegment, ...]

    @property
    def segment_starts(self) -> tuple[int, ...]:
        return tuple(segment.start for segment in self.segments)


class StreamingWeightTrajectoryCompressor:
    """Compress adapted model weights as a streaming piecewise-linear trajectory."""

    def __init__(
        self,
        residual_threshold: float | None,
        *,
        auto_multiplier: float = 2.0,
    ) -> None:
        if residual_threshold is not None and residual_threshold < 0:
            raise ValueError("residual_threshold must be non-negative")
        if auto_multiplier <= 0:
            raise ValueError("auto_multiplier must be positive")
        self.residual_threshold = residual_threshold
        self.auto_multiplier = auto_multiplier
        self._fitter: _StreamingLinearFitter | None = None
        self._dimension: int | None = None
        self._current_start: int | None = None
        self._latest_step: int | None = None
        self._closed_segments: list[WeightTrajectorySegment] = []
        self._latest_residual = 0.0

    @staticmethod
    def _make_segment(
        fitter: _StreamingLinearFitter,
        start: int,
        end: int,
    ) -> WeightTrajectorySegment:
        slope, intercept = fitter.coefficients()
        return WeightTrajectorySegment(
            start,
            end,
            slope.astype(np.float32),
            intercept.astype(np.float32),
        )

    def _current_segments(self) -> tuple[WeightTrajectorySegment, ...]:
        segments = list(self._closed_segments)
        if (
            self._fitter is not None
            and self._current_start is not None
            and self._latest_step is not None
        ):
            segments.append(
                self._make_segment(
                    self._fitter,
                    self._current_start,
                    self._latest_step,
                )
            )
        return tuple(segments)

    def update(self, step: int, weight_vector: np.ndarray) -> CompressionUpdate:
        """Add one adapted model and update the compressed weight trajectory."""
        weight_vector = np.asarray(weight_vector, dtype=np.float64).reshape(-1)
        if weight_vector.size == 0:
            raise ValueError("weight_vector cannot be empty")
        if self._latest_step is not None and step <= self._latest_step:
            raise ValueError("steps must be strictly increasing")

        if self._fitter is None:
            self._dimension = weight_vector.size
            self._fitter = _StreamingLinearFitter(weight_vector.size)
            self._current_start = step
            self._latest_step = step
            self._fitter.add(0.0, weight_vector)
            return self._build_update(step, True)

        if weight_vector.size != self._dimension:
            raise ValueError(
                f"expected dimension {self._dimension}, got {weight_vector.size}"
            )

        previous_fitter = self._fitter.copy()
        previous_step = self._latest_step
        assert self._current_start is not None and previous_step is not None
        self._fitter.add(float(step - self._current_start), weight_vector)
        self._latest_step = step
        self._latest_residual = self._fitter.squared_residual()

        if (
            self.residual_threshold is None
            and self._fitter.count >= 3
            and self._latest_residual > 0
        ):
            self.residual_threshold = max(
                self._latest_residual * self.auto_multiplier,
                1e-12,
            )

        starts_new_segment = (
            self.residual_threshold is not None
            and self._latest_residual > self.residual_threshold
        )
        if starts_new_segment:
            self._closed_segments.append(
                self._make_segment(
                    previous_fitter,
                    self._current_start,
                    previous_step,
                )
            )
            self._fitter = _StreamingLinearFitter(weight_vector.size)
            self._fitter.add(0.0, weight_vector)
            self._current_start = step
            self._latest_step = step
            self._latest_residual = 0.0

        return self._build_update(step, starts_new_segment)

    def _build_update(
        self,
        step: int,
        starts_new_segment: bool,
    ) -> CompressionUpdate:
        return CompressionUpdate(
            step=step,
            starts_new_segment=starts_new_segment,
            latest_residual=self._latest_residual,
            threshold=self.residual_threshold,
            segments=self._current_segments(),
        )

    def finalize(self) -> tuple[WeightTrajectorySegment, ...]:
        return self._current_segments()

