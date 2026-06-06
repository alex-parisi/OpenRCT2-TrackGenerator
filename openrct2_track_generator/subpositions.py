"""
Vehicle subposition tables, sampled from the section curves.

A subposition is a fine-grained point a vehicle occupies as it travels a track
piece: a position plus the orientation it sits at there. Because the section curves
already give a moving frame, the table is just the curve sampled at fixed intervals
with Tait-Bryan angles read off the frame.

NOTE (open question): OpenRCT2's in-game subposition encoding uses discretized yaw
(0-31) and pitch/bank enum slots. This module emits **float radians** plus raw
curve-space positions as an intermediate; the discretization to the game's encoding
is deferred until the target format is confirmed. The output is therefore written as
a sidecar JSON rather than embedded in object.json.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import Track, TrackSection

__all__ = [
    "DEFAULT_SUBPOSITION_SAMPLES",
    "Subposition",
    "build_subposition_data",
    "sample_subpositions",
]

# Number of samples per section. OpenRCT2 uses denser tables; a fixed modest count
# is enough for the vertical slice and keeps the sidecar readable.
DEFAULT_SUBPOSITION_SAMPLES: int = 16


@dataclass(frozen=True, slots=True)
class Subposition:
    """One sampled point along a track piece.

    ``position`` is in curve-space OBJ units; ``yaw``/``pitch``/``roll`` are radians
    derived from the moving frame (yaw in the horizontal plane, pitch the rise, roll
    the bank).
    """

    distance: float
    position: tuple[float, float, float]
    yaw: float
    pitch: float
    roll: float


def sample_subpositions(
    section: TrackSection, *, num_samples: int = DEFAULT_SUBPOSITION_SAMPLES
) -> list[Subposition]:
    """Sample ``section``'s curve at ``num_samples`` evenly spaced arc-distances."""
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    d = np.linspace(0.0, section.length, num_samples)
    tp = section.curve(d)
    yaw = np.arctan2(tp.tangent[:, 0], tp.tangent[:, 2])
    pitch = np.arcsin(np.clip(tp.tangent[:, 1], -1.0, 1.0))
    # Bank: how far the up-axis has rolled out of vertical. Zero for unbanked frames
    # (normal_y = 1, binormal_y = 0 -> atan2(0, 1) = 0).
    roll = np.arctan2(-tp.binormal[:, 1], tp.normal[:, 1])
    return [
        Subposition(
            distance=float(d[i]),
            position=(
                float(tp.position[i, 0]),
                float(tp.position[i, 1]),
                float(tp.position[i, 2]),
            ),
            yaw=float(yaw[i]),
            pitch=float(pitch[i]),
            roll=float(roll[i]),
        )
        for i in range(num_samples)
    ]


def build_subposition_data(
    track: Track, *, num_samples: int = DEFAULT_SUBPOSITION_SAMPLES
) -> dict[str, Any]:
    """Build the JSON-able subposition sidecar for every section in ``track``."""
    sections: list[dict[str, Any]] = []
    for section in track.sections:
        subs = sample_subpositions(section, num_samples=num_samples)
        sections.append(
            {
                "section": section.name,
                "length": section.length,
                "subpositions": [
                    {
                        "distance": s.distance,
                        "x": s.position[0],
                        "y": s.position[1],
                        "z": s.position[2],
                        "yaw": s.yaw,
                        "pitch": s.pitch,
                        "roll": s.roll,
                    }
                    for s in subs
                ],
            }
        )
    return {
        "id": track.id,
        "units_per_tile": track.units_per_tile,
        "angle_units": "radians",
        "sections": sections,
    }
