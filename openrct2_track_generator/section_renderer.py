"""
Render one track section into its four cardinal-view sprites.

Each section is deformed once (the new track-specific step), then handed to the
existing X7 render path unchanged: add the deformed mesh with an identity transform
and render the four ``VIEWS`` OpenRCT2 cycles through as the park view rotates.
"""

import numpy as np
from openrct2_x7_renderer.ray_trace import VIEWS, Context
from openrct2_x7_renderer.types import IndexedImage

from .deform import deform_mesh
from .types import Track, TrackSection

__all__ = ["render_section"]


def render_section(track: Track, section: TrackSection, context: Context) -> list[IndexedImage]:
    """Deform ``section`` from the track tile mesh and render its four views."""
    deformed = deform_mesh(
        track.meshes[track.track_mesh_index], section, flat_shaded=track.flat_shaded
    )
    builder = context.begin_render()
    # The deformation is already baked into the mesh, so the placement is identity.
    builder.add_model(deformed, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0)
    scene = builder.finalize()
    try:
        return [scene.render_view(view) for view in VIEWS]
    finally:
        scene.end_render()
