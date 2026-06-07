# track_renderer_addon

The Blender 4.2+ add-on (extension). It is the UI + scene adapter only: the whole
pipeline (config validation, deformation, rendering, sprite-manifest export) lives
in the bundled [`openrct2_track_generator`](../openrct2_track_generator/) and
[`openrct2-x7-renderer`](https://pypi.org/project/openrct2-x7-renderer/) wheels.
This package reads the Blender scene, hands the core an in-memory config dict, a
`Mesh` list, and a `special_models` map, and surfaces the result in the viewport.

It ships as a separate extension (`id = "openrct2tg"`) from the vehicle
(`openrct2vg`) and scenery (`openrct2sg`) add-ons, so all three can be installed
at once.

## How it works

1. **Properties** (`props.py`): native Blender `PropertyGroup`s store the whole
   track in the `.blend` file: track-wide settings on `scene.tg_track` (id, ride
   type, scale, z-offset, lift, supports, brake length, and the **sections** list),
   per-object role on `object.tg_object` (Track / Mask / Special-Support / Ignore,
   plus the model key for specials), per-material region/shading on
   `material.tg_material`. The section and special-model enum item lists are
   sourced from the installed `openrct2_track_generator` package, so the UI can
   never offer a value the core would reject.
2. **Panels** (`panels.py`): draw those properties in the 3D Viewport **OpenRCT2**
   sidebar tab: `TG_PT_track` (track-wide) and `TG_PT_object_view3d` (active object
   + its materials), plus the `UIList`s for the sections (`TG_UL_sections`) and
   custom lights (`TG_UL_lights`). The selected-object panel shares a `bl_idname`
   with the vehicle/scenery add-ons' parent so the extensions stack under one
   header.
3. **Scene adapter** (`scene_to_track.py`): the `bpy → Track` bridge. It bakes
   each object's world transform into an in-memory `Mesh` (no OBJ files written),
   classifies objects by their role, and applies maketrack's `rotate_y(-90°)` load
   rotation so meshes authored with the rail along Blender +X become +Z
   along-track for the deform. It builds the config dict the core's `build_track`
   expects.
4. **Operators** (`operators.py`): `tg.test_render` renders the listed sections and
   loads the first into an Image Editor for fast iteration; `tg.export` renders
   every section on a background thread (spinner in the status bar) and writes the
   sprite manifest + indexed PNGs (+ subposition sidecar) into a chosen directory.
   `tg.section_add`/`tg.section_remove` and `tg.light_add`/`tg.light_remove`
   manage the `UIList`s. All call the same core `build_track` → render → export
   path the CLI uses.

`__init__.py` registers props → operators → panels in that order (panels draw
properties, so the property groups must exist first).

## Authoring a track

- Model the **track tile** (the rail) with the along-track axis on Blender **+X**,
  spanning one tile; set its role to **Track**.
- Optionally add the **occlusion mask** plate (role **Mask**, material flagged
  *Visible Mask*) to enable the front/behind carve for self-overlapping pieces.
- Optionally add **brake / booster / support** meshes (role **Special / Support**)
  and pick which model each is — they're placed by the same selectors the CLI uses.
- Pick the **sections** to generate and the track-wide settings, then
  **Test Render** or **Export Sprites**.

## Coordinate convention

OBJ space is +X forward, +Y up, +Z right. A Blender vertex `(bx, by, bz)` maps to
OBJ `(bx, bz, -by)` via the shared `BASIS` matrix (a proper rotation, det = +1, so
winding is preserved); the adapter then applies `rotate_y(-90°)` so the rail's
along-track +X becomes the deform's +Z. 1 tile = `units_per_tile` OBJ units.

## Packaging model

Blender extensions run in an isolated Python environment and install **only** the
wheels listed in `blender_manifest.toml`; pip is never consulted at install time.
So everything the add-on imports must be vendored as a wheel for every
platform × Python combination Blender ships. Three kinds of wheel are bundled
under `wheels/`:

| Wheel | Source | Variants |
|---|---|---|
| `openrct2_x7_renderer` | PyPI (Embree-vendored native extension) | per platform × CPython 3.11/3.13 |
| `numpy`, `pillow`, `pyyaml`, `openrct2_objectcommon` | PyPI | per platform × CPython 3.11/3.13 (deps); one `py3-none-any` (objectcommon) |
| `openrct2_trackgenerator` | this repo (`uv build --wheel`, pure Python) | one `py3-none-any` for all targets |

## Building the extension

```bash
# Local single-platform build for the Blender on THIS machine:
uv run python scripts/build_plugin_local.py

# Refresh the committed wheels/ + manifest for all target platforms:
uv build --wheel                              # build the front-end wheel first
uv run python scripts/collect_wheels.py       # download deps, regenerate manifest
blender --command extension build             # zip the extension
```

`build_plugin_local.py` stages everything in a temp dir and never touches the
committed `wheels/` or `blender_manifest.toml`.

## Source layout

```
track_renderer_addon/
├── __init__.py            # register/unregister (props -> operators -> panels)
├── blender_manifest.toml  # extension manifest (id, version, platforms, wheels)
├── props.py               # PropertyGroups: track/object/material data + section/light lists
├── panels.py              # 3D Viewport OpenRCT2 sidebar panels + UILists
├── operators.py           # tg.test_render + threaded tg.export + section/light ops
├── scene_to_track.py      # bpy -> Mesh adapter + config-dict builder
└── wheels/                # vendored wheels (regenerated by collect_wheels.py)
```

## License

GPL-3.0-or-later. The bundled wheels carry Embree + TBB (Apache-2.0); their
license texts ship alongside in the extension zip.
