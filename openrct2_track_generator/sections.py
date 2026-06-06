"""
The registry binding section names to their geometry.

This is the seam that scales the generator to the full RCTGen catalogue: adding a
section is a one-line :class:`TrackSection` entry referencing a curve from
:mod:`~openrct2_track_generator.curves`. The vertical slice registers six. Flags and
arc lengths are copied from ``track_sections.cpp``'s ``const track_section_t`` table.
"""

from . import curves
from .constants import TrackFlag
from .types import TrackSection

__all__ = ["SECTION_REGISTRY", "resolve_section"]

SECTION_REGISTRY: dict[str, TrackSection] = {
    "flat": TrackSection("flat", curves.flat_curve, curves.FLAT_LENGTH),
    "flat_to_gentle": TrackSection(
        "flat_to_gentle", curves.flat_to_gentle_curve, curves.FLAT_TO_GENTLE_LENGTH
    ),
    "gentle": TrackSection("gentle", curves.gentle_curve, curves.GENTLE_LENGTH),
    "gentle_to_steep": TrackSection(
        "gentle_to_steep",
        curves.gentle_to_steep_curve,
        curves.GENTLE_TO_STEEP_LENGTH,
        flags=TrackFlag.ALT_PREFER_ODD,
    ),
    "steep": TrackSection(
        "steep", curves.steep_curve, curves.STEEP_LENGTH, flags=TrackFlag.ALT_INVERT
    ),
    "small_turn_left": TrackSection(
        "small_turn_left",
        curves.small_turn_left_curve,
        curves.SMALL_TURN_LENGTH,
        flags=TrackFlag.EXIT_90_DEG_LEFT,
    ),
}


def resolve_section(name: str) -> TrackSection:
    """Look up a section by name, raising ``KeyError`` with the known names if absent."""
    try:
        return SECTION_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(SECTION_REGISTRY))
        raise KeyError(f'Unknown track section "{name}" (known: {known})') from None
