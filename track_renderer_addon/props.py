"""Blender PropertyGroups for the track add-on.

Mirrors the config the core ``build_track`` consumes (see scene_to_track.py),
expressed as native Blender properties so a track object is authored in the UI:
a track-tile mesh, an occlusion-mask mesh, optional special/support meshes, and
the list of sections to generate. Uses a ``tg_`` prefix so this add-on can
coexist with the vehicle (``vg_``) and scenery (``vgs_``) add-ons.

NOTE: no ``from __future__ import annotations``; PEP 563 would stringify the
``prop: SomeProperty(...)`` definitions and break Blender registration.
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Material, Object, PropertyGroup, Scene
from openrct2_object_common.blender.props import (
    SCALE_PRESET_ITEMS,
    SharedLight,
    scale_preset_update,
)
from openrct2_track_generator.constants import (
    SPECIAL_MODEL_KEY,
    SUPPORT_BANK_KEY,
    SUPPORT_BASE_KEY,
)
from openrct2_track_generator.sections import SECTION_REGISTRY
from openrct2_x7_renderer.constants import TILE_SIZE


def _scale_preset_update(self, _context):
    scale_preset_update(self, _context)


# Same material-region scheme as the vehicle/scenery add-ons; the rail is usually
# REMAP1 (recoloured by the track's main colour) and the lift rail uses CHAIN.
MATERIAL_REGION_ITEMS = [
    ("NONE", "None", "Plain shaded colour"),
    ("REMAP1", "Remap 1 (primary colour)", "Recoloured by the track's primary colour"),
    ("REMAP2", "Remap 2 (secondary)", "Recoloured by the secondary colour"),
    ("REMAP3", "Remap 3 (tertiary)", "Recoloured by the tertiary colour"),
    ("GREYSCALE", "Greyscale", "Greyscale shading region"),
    ("PEEP", "Peep", "Peep region"),
    ("CHAIN", "Chain", "Lift-hill chain region"),
]

# A scene object's role in the track object. The track tile is deformed along the
# section curves; the mask plate drives occlusion carving; SPECIAL meshes are the
# named brake/booster/support models placed by their selector.
OBJECT_ROLE_ITEMS = [
    ("TRACK", "Track", "The deformable track-tile mesh (the rail)"),
    ("MASK", "Mask", "The occlusion-mask plate (use a VisibleMask material)"),
    ("SPECIAL", "Special / Support", "A named brake / booster / support model"),
    ("IGNORE", "Ignore", "Not part of the track object"),
]

# The named auxiliary models the core resolves by key (brake/booster specials +
# the support base and bank posts + the rigid inversion supports). A SPECIAL
# object is matched to the core by picking one of these.
_SPECIAL_MODEL_NAMES = sorted(
    {*SPECIAL_MODEL_KEY.values(), SUPPORT_BASE_KEY, *SUPPORT_BANK_KEY}
)
SPECIAL_MODEL_ITEMS = [(n, n, "") for n in _SPECIAL_MODEL_NAMES]

# Every section the generator knows; a track lists the ones it wants to emit.
SECTION_ITEMS = [(n, n, "") for n in sorted(SECTION_REGISTRY)]


class TGMaterialSettings(PropertyGroup):
    region: EnumProperty(
        name="Region",
        description="How OpenRCT2 treats this material's pixels",
        items=MATERIAL_REGION_ITEMS,
        default="REMAP1",
    )
    is_mask: BoolProperty(name="Mask", default=False)
    is_visible_mask: BoolProperty(
        name="Visible Mask",
        description="Rendered into the occlusion silhouette (use on the mask plate)",
        default=False,
    )
    no_ao: BoolProperty(name="No Ambient Occlusion", default=False)
    edge: BoolProperty(name="Edge AA", default=False)
    dark_edge: BoolProperty(name="Dark Edge AA", default=False)
    no_bleed: BoolProperty(name="No Bleed", default=False)
    flat_shaded: BoolProperty(name="Flat Shaded", default=False)
    texture: PointerProperty(
        name="Texture",
        description="Optional image; must be saved to disk (its file is read at export)",
        type=bpy.types.Image,
    )
    use_color_override: BoolProperty(
        name="Override Color",
        description="Use the color below instead of the shader's Base Color",
        default=False,
    )
    diffuse_color: FloatVectorProperty(
        name="Color", subtype="COLOR", size=3, min=0.0, max=1.0, default=(0.8, 0.8, 0.8)
    )
    specular_intensity: FloatProperty(
        name="Specular Intensity", default=0.5, min=0.0, soft_max=1.0
    )
    specular_exponent: FloatProperty(
        name="Specular Exponent", default=50.0, min=1.0, soft_max=256.0
    )
    use_specular_tint: BoolProperty(name="Tint Highlight", default=False)
    specular_tint: FloatVectorProperty(
        name="Specular Tint", subtype="COLOR", size=3, min=0.0, max=1.0, default=(1.0, 1.0, 1.0)
    )


class TGObjectSettings(PropertyGroup):
    role: EnumProperty(
        name="Role",
        description="This object's part in the track",
        items=OBJECT_ROLE_ITEMS,
        default="TRACK",
    )
    special_model: EnumProperty(
        name="Model",
        description="Which named brake/booster/support model this mesh is",
        items=SPECIAL_MODEL_ITEMS,
        default="brake",
    )


class TGSection(PropertyGroup):
    # NB: not "name" — PropertyGroup reserves a built-in ``name`` StringProperty that
    # would shadow this EnumProperty (the picker would silently become a text field).
    section: EnumProperty(name="Section", items=SECTION_ITEMS, default="flat")


TGLight = SharedLight


class TGTrackSettings(PropertyGroup):
    # --- Identity ----------------------------------------------------------
    id: StringProperty(
        name="Object ID",
        description="Unique id, e.g. openrct2tg.track.my_coaster (avoid vanilla ids)",
        default="openrct2tg.track.my_coaster",
    )
    name: StringProperty(name="Name", default="My Track")
    description: StringProperty(name="Description", default="")
    authors: StringProperty(name="Authors", description="Comma-separated", default="")
    version: StringProperty(name="Version", default="1.0")
    ride_type: StringProperty(name="Ride Type", default="steel_roller_coaster")
    preview: StringProperty(
        name="Preview Image", description="Path to a preview image", subtype="FILE_PATH", default=""
    )

    # --- Geometry scale ----------------------------------------------------
    scale_preset: EnumProperty(
        name="Scale",
        description="How many OBJ units map to one OpenRCT2 tile",
        items=SCALE_PRESET_ITEMS,
        default="REALISTIC",
        update=_scale_preset_update,
    )
    units_per_tile: FloatProperty(
        name="Units / Tile", default=TILE_SIZE, min=0.01, soft_max=16.0
    )
    flat_shaded: BoolProperty(name="Flat Shaded", default=False)

    # --- Track placement ---------------------------------------------------
    # z_offset is in 1/8-clearance-height units (maketrack's track z_offset).
    z_offset: FloatProperty(
        name="Z Offset",
        description="Vertical offset in 1/8-clearance-height units (e.g. 5)",
        default=0.0,
    )

    # --- Lift hill ---------------------------------------------------------
    has_lift: BoolProperty(
        name="Lift Hill",
        description="Overlay the chain pattern and emit all 4 directions",
        default=False,
    )

    # --- Supports (tile fractions, scaled by units/tile in the core) -------
    has_supports: BoolProperty(
        name="Supports",
        description="Emit a per-tile base + support posts (needs support_* models)",
        default=False,
    )
    support_spacing: FloatProperty(
        name="Support Spacing", description="Post spacing in tiles", default=1.0, min=0.05
    )
    pivot: FloatProperty(
        name="Support Pivot", description="Post pivot height in tiles", default=0.0
    )
    brake_length: FloatProperty(
        name="Brake Length",
        description="Brake-mechanism tile length in tiles",
        default=1.0,
        min=0.05,
    )

    # --- Sections ----------------------------------------------------------
    sections: CollectionProperty(type=TGSection)
    section_index: IntProperty(default=0)

    # --- Custom lighting ---------------------------------------------------
    lights: CollectionProperty(type=TGLight)
    light_index: IntProperty(default=0)
    show_lights: BoolProperty(name="Custom Lighting", default=False)


# TGLight (= SharedLight) is registered cooperatively, NOT in _CLASSES: Blender shares
# the bundled openrct2_objectcommon wheel across the OpenRCT2 add-ons, so SharedLight is
# one class object — whichever add-on loads first registers it, the rest must skip it
# (else "already registered as a subclass 'SharedLight'"). Mirrors the shared parent
# panel guard in panels.py.
_CLASSES = (
    TGMaterialSettings,
    TGObjectSettings,
    TGSection,
    TGTrackSettings,
)

_shared_light_owned = False


def _register_shared_light():
    """Register SharedLight unless another OpenRCT2 add-on already did.

    Blender shares the bundled wheel, so ``TGLight`` is the very class object the
    other add-ons register; ``is_registered`` is the reliable cross-add-on check
    (the class is not exposed as ``bpy.types.SharedLight``).
    """
    global _shared_light_owned
    if not TGLight.is_registered:
        bpy.utils.register_class(TGLight)
        _shared_light_owned = True


def _unregister_shared_light():
    """Drop SharedLight only if this add-on was the one that registered it."""
    global _shared_light_owned
    if _shared_light_owned:
        bpy.utils.unregister_class(TGLight)
        _shared_light_owned = False


def register():
    # SharedLight must exist before TGTrackSettings' CollectionProperty(type=TGLight).
    _register_shared_light()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    Scene.tg_track = PointerProperty(type=TGTrackSettings)
    Object.tg_object = PointerProperty(type=TGObjectSettings)
    Material.tg_material = PointerProperty(type=TGMaterialSettings)


def unregister():
    del Material.tg_material
    del Object.tg_object
    del Scene.tg_track
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    _unregister_shared_light()
