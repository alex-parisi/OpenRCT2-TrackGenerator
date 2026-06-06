"""
Lift-hill chain overlay, ported from RCTGen's ``sprites.cpp`` + ``apply_lift``.

On a lift hill the track rails carry a moving chain. maketrack renders this by
stamping a small repeating chain pattern over the rail pixels: every pixel in the
remappable track-colour range (palette indices 1-3) is replaced by the pattern,
tiled across the sprite and aligned by the pattern's offset. There is one pattern
per view direction (the chain points along the track), which is why a lift-enabled
flat piece needs four sprites instead of two.
"""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from openrct2_x7_renderer.types import IndexedImage

__all__ = ["CHAIN_PATTERNS", "ChainPattern", "apply_lift"]


@dataclass(frozen=True, slots=True)
class ChainPattern:
    """One direction's chain stamp: its size, draw offset, and (height, width) pixels."""

    x_offset: int
    y_offset: int
    pixels: NDArray[np.uint8]  # (height, width) palette indices

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def _chain(width: int, height: int, x_offset: int, y_offset: int, flat: list[int]) -> ChainPattern:
    return ChainPattern(x_offset, y_offset, np.array(flat, dtype=np.uint8).reshape(height, width))


# sprites.cpp: flat_pixels split into four 3x6 stamps.
_FLAT_PIXELS = [
    1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 2, 3, 1, 2, 3, 1,
    1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2,
    1, 3, 2, 1, 3, 2, 2, 1, 3, 2, 1, 3, 3, 2, 1, 3, 2, 1,
    1, 3, 2, 1, 3, 2, 3, 2, 1, 3, 2, 1, 2, 1, 3, 2, 1, 3,
]
FLAT_CHAIN = [
    _chain(3, 6, 0, -2, _FLAT_PIXELS[0:18]),
    _chain(3, 6, 0, -1, _FLAT_PIXELS[18:36]),
    _chain(3, 6, 0, 0, _FLAT_PIXELS[36:54]),
    _chain(3, 6, -1, 0, _FLAT_PIXELS[54:72]),
]

# sprites.cpp: gentle_pixels split into 6x3 / 3x1 / 3x1 / 6x3 stamps.
_GENTLE_PIXELS = [
    1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1,
    1, 2, 3,
    3, 2, 1,
    2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 3, 3, 2, 2, 1, 1,
]
GENTLE_CHAIN = [
    _chain(6, 3, -3, -1, _GENTLE_PIXELS[0:18]),
    _chain(3, 1, 1, 0, _GENTLE_PIXELS[18:21]),
    _chain(3, 1, 1, 0, _GENTLE_PIXELS[21:24]),
    _chain(6, 3, 0, -1, _GENTLE_PIXELS[24:42]),
]

# sprites.cpp: flat_diag_chain — a 3-pixel stamp [1,2,3] / [3,2,1] as 3x1 or 1x3.
_FLAT_DIAG_A = [1, 2, 3]
_FLAT_DIAG_B = [3, 2, 1]
FLAT_DIAG_CHAIN = [
    _chain(3, 1, -2, 0, _FLAT_DIAG_A),
    _chain(1, 3, 0, -2, _FLAT_DIAG_A),
    _chain(3, 1, -1, 0, _FLAT_DIAG_B),
    _chain(1, 3, 0, -1, _FLAT_DIAG_B),
]

CHAIN_PATTERNS: dict[str, list[ChainPattern]] = {
    "flat": FLAT_CHAIN,
    "gentle": GENTLE_CHAIN,
    "flat_diag": FLAT_DIAG_CHAIN,
}


def apply_lift(image: IndexedImage, pattern: ChainPattern) -> IndexedImage:
    """Overlay ``pattern`` onto ``image``'s rail pixels (palette indices 1-3).

    The pattern tiles across the sprite (modulo its size) aligned by the draw offsets,
    matching ``track.cpp::apply_lift``. Non-rail pixels are left unchanged.
    """
    h, w = image.pixels.shape
    cols = np.arange(w)
    rows = np.arange(h)
    xi = np.mod(cols + image.x_offset - pattern.x_offset, pattern.width)
    yi = np.mod(rows + image.y_offset - pattern.y_offset, pattern.height)
    stamped = pattern.pixels[yi[:, None], xi[None, :]]
    rail = (image.pixels >= 1) & (image.pixels <= 3)
    return replace(image, pixels=np.where(rail, stamped, image.pixels).astype(np.uint8))
