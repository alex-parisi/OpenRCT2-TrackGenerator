"""
Usage:
    openrct2-track-generator [--test|--skip-render] <input.json|.yaml>
    python -m openrct2_track_generator [--test|--skip-render] <input.json|.yaml>
"""

import argparse
import sys
from typing import Any

from openrct2_object_common.cli import make_context, output_directory_of, run_cli
from openrct2_x7_renderer.types import Light

from .exporter import export_track_test, export_tracks
from .loader import build_tracks


def _render(args: argparse.Namespace, root: dict[str, Any], lights: list[Light]) -> None:
    # run_cli already parsed the config into `root`; build the track(s) straight from it
    # (one Track, or several variants when the config has a `tracks` array).
    tracks = build_tracks(root)
    context = make_context(lights, tracks[0].units_per_tile, args.test, root)
    if args.test:
        for track in tracks:
            export_track_test(track, context)
    else:
        export_tracks(tracks, context, output_directory_of(root), skip_render=args.skip_render)


def main(argv: list[str] | None = None) -> int:
    return run_cli("openrct2-track-generator", argv, _render)


if __name__ == "__main__":
    sys.exit(main())
