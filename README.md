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
* `groups.py` — section-group expansion (port of `write_track_type`).
* `offsets.py` — `special_end_offsets` endpoint-offset table (port of `track.cpp`).
* `subpositions.py` — vehicle subposition tables in OpenRCT2's discretized sprite encoding
  (port of `subposition.cpp`); the rotation tables are embedded in `_sprite_rotations.py`.

## Status: maketrack parity

All **170** track sections are implemented (name-for-name with `track_sections.cpp`),
alongside the rest of `maketrack`'s surface:

* **Section groups** — list maketrack group names (`turns`, `gentle_slopes`,
  `banked_sloped_turns`, …) in `sections`; they expand to the constituent pieces. Individual
  section names still work too.
* **Masks**, **supports** (base + banked posts), **lift-hill chains**, and
  **special-mechanism** models (brakes/boosters/inversions).
* **`special_end_offsets`** endpoint smoothing via an `offsets` table.
* **Separate ties** (`separate_tie` / `tie_at_boundary`).
* **`spritefile_in`/`spritefile_out` merge** — append rendered sprites into an existing
  manifest (the fixed-global-image-index workflow).
* **Multiple track variants per file** — a `tracks` array whose entries inherit the shared
  config and override per-variant keys; each variant's `suffix` keeps sprite filenames
  distinct in one merged manifest.
* **Subposition tables** — `subpositions.py` emits OpenRCT2's discretized yaw/pitch/bank
  sprite indices (not raw radians).

`maketrack` and the separate `subposition` tool are both ported. The X7 renderer differs
from RCTGen's IsoRender, so output is structurally — not pixel — identical.

## Output format

OpenRCT2 has **no track object type** — track sprites are baked graphics referenced by
hardcoded global image indices, compiled from a *sprite manifest* via
`openrct2 sprite build`. So, like the original `maketrack`, this generator emits
palette-indexed PNGs plus a manifest array of `{path, x, y, palette: "keep"}` entries
(addressed by array order), not a `.parkobj`. A full run writes:

```
<output>/<id>.sprites.json                      # the sprite manifest (or spritefile_out)
<output>/images/<section><suffix>_<view>_<sub>.png  # palette-indexed sprites
<output>/<id>.subpositions.json                 # discretized subposition tables (sidecar)
```

The image directory (`sprite_directory`), an existing manifest to merge into
(`spritefile_in`) and the output manifest (`spritefile_out`) are all configurable; `suffix`
is empty unless set per track variant.

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
