"""
Usage:
    openrct2-track-generator [--test|--skip-render] <input.json|.yaml>
    python -m openrct2_track_generator [--test|--skip-render] <input.json|.yaml>
"""

import argparse
import sys
from typing import Any

from openrct2_object_common.cli import make_context, output_directory_of, run_cli
from openrct2_object_common.config import load_meshes, load_preview
from openrct2_x7_renderer.types import Light

from .exporter import export_track, export_track_test
from .loader import build_track


def _render(args: argparse.Namespace, root: dict[str, Any], lights: list[Light]) -> None:
    # run_cli already parsed the config into `root`; build straight from it.
    track = build_track(root, load_meshes(root), load_preview(root))
    context = make_context(lights, track.units_per_tile, args.test, root)
    if args.test:
        export_track_test(track, context)
    else:
        export_track(track, context, output_directory_of(root), skip_render=args.skip_render)


def main(argv: list[str] | None = None) -> int:
    return run_cli("openrct2-track-generator", argv, _render)


if __name__ == "__main__":
    sys.exit(main())
