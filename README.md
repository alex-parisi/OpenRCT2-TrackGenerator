# OpenRCT2-TrackGenerator

Author and export OpenRCT2 roller-coaster **track** sprites from meshes, rendering
through [`openrct2-x7-renderer`](https://github.com/alex-parisi/OpenRCT2-X7-Renderer)
and packaging through
[`OpenRCT2-ObjectCommon`](https://github.com/alex-parisi/OpenRCT2-ObjectCommon).

This is the third generator in the `OpenRCT2-Tools` family, alongside the
[Vehicle](https://github.com/alex-parisi/OpenRCT2-VehicleGenerator) and
[Scenery](https://github.com/alex-parisi/OpenRCT2-SceneryGenerator) generators. It is a
Python port of the track-rendering half of the original RCTGen `maketrack` executable.

## What makes track generation different

The vehicle and scenery generators only apply **rigid** transforms to whole meshes. A
track piece, by contrast, is a single tile-length mesh **deformed along a parametric
curve** (a `flat`, a `gentle` slope, a `small_turn_left`, …) before it is rendered. The
deformation maps each vertex's local `z` onto an arc-distance along the curve and places
its `x`/`y` in the curve's cross-section frame (`position + normal·y + binormal·x`). The
deformed mesh is then a drop-in for the existing X7 render path.

* `curves.py` — vectorized curve library (port of `track_sections.cpp`).
* `deform.py` — vectorized mesh deformation (port of `track.cpp`'s `track_transform`).
* `subpositions.py` — vehicle subposition tables, sampled from the same curves.

## Status: vertical slice

This milestone ports six representative sections end-to-end to prove the pipeline:
`flat`, `flat_to_gentle`, `gentle`, `gentle_to_steep`, `steep`, `small_turn_left`.
Broad section coverage, masks, supports, and special inversions are later milestones.

## Output format

OpenRCT2 has **no track object type** — track sprites are baked graphics referenced by
hardcoded global image indices, compiled from a *sprite manifest* via
`openrct2 sprite build`. So, like the original `maketrack`, this generator emits
palette-indexed PNGs plus a manifest array of `{path, x, y, palette: "keep"}` entries
(addressed by array order), not a `.parkobj`. A full run writes:

```
<output>/<id>.sprites.json        # the sprite manifest
<output>/images/<section>_<n>.png # palette-indexed sprites (n = view 1..4)
<output>/<id>.subpositions.json   # vehicle subposition tables (sidecar)
```

Compile the sprites into a graphics file with OpenRCT2's CLI:

```bash
openrct2 sprite build mytrack.dat <id>.sprites.json
```

## Usage

```bash
# preview (single-viewpoint, zoomed) into ./test
openrct2-track-generator --test examples/demo_track.yaml

# full render -> sprite manifest + PNGs (+ subpositions sidecar)
openrct2-track-generator examples/demo_track.yaml
```
