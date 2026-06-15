"""UI panels for the track add-on: scene settings (3D View N-panel) + per-object
role + per-material region."""

import bpy
from bpy.types import Panel, UIList
from openrct2_object_common.blender.bake import draw_bake


class TG_UL_sections(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        row = layout.row(align=True)
        row.label(text="", icon="IPO_BEZIER")
        row.prop(item, "section", text="")


class TG_UL_lights(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        row = layout.row(align=True)
        row.label(text="", icon="LIGHT")
        row.prop(item, "type", text="")
        row.prop(item, "strength", text="")


class TG_PT_track(Panel):
    bl_label = "OpenRCT2 Track"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRCT2"

    def draw(self, context):
        layout = self.layout
        tt = context.scene.tg_track

        layout.prop(tt, "scale_preset")
        if tt.scale_preset == "CUSTOM":
            layout.prop(tt, "units_per_tile")
        box = layout.box()
        box.label(text="Identity", icon="INFO")
        box.prop(tt, "id")
        box.prop(tt, "name")
        box.prop(tt, "description")
        box.prop(tt, "authors")
        box.prop(tt, "version")
        box.prop(tt, "ride_type")

        box = layout.box()
        box.label(text="Dither", icon="MOD_NOISE")
        box.prop(tt, "dither")
        box.prop(tt, "dither_stability")

        box = layout.box()
        box.label(text="Track", icon="TOOL_SETTINGS")
        box.prop(tt, "z_offset")
        box.prop(tt, "flat_shaded")
        box.prop(tt, "has_lift")

        box = layout.box()
        box.prop(tt, "has_supports", icon="MOD_BUILD")
        if tt.has_supports:
            box.prop(tt, "support_spacing")
            box.prop(tt, "pivot")
            box.label(text="Needs support_base / support_* meshes.", icon="INFO")
        box.prop(tt, "brake_length")

        box = layout.box()
        box.label(text="Sections", icon="IPO_BEZIER")
        row = box.row()
        row.template_list("TG_UL_sections", "", tt, "sections", tt, "section_index", rows=4)
        col = row.column(align=True)
        col.operator("tg.section_add", icon="ADD", text="")
        col.operator("tg.section_remove", icon="REMOVE", text="")
        if not tt.sections:
            box.label(text="No sections - add at least one.", icon="ERROR")

        box = layout.box()
        row = box.row()
        row.prop(
            tt,
            "show_lights",
            icon="TRIA_DOWN" if tt.show_lights else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="", icon="LIGHT_SUN")
        if tt.show_lights:
            row = box.row()
            row.template_list("TG_UL_lights", "", tt, "lights", tt, "light_index", rows=3)
            col = row.column(align=True)
            col.operator("tg.light_add", icon="ADD", text="")
            col.operator("tg.light_remove", icon="REMOVE", text="")
            if tt.lights:
                light = tt.lights[tt.light_index]
                sub = box.column()
                sub.prop(light, "type")
                sub.prop(light, "shadow")
                sub.prop(light, "direction")
                sub.prop(light, "strength")
            else:
                box.label(text="No lights - using the default rig.", icon="INFO")

        layout.prop(tt, "preview")

        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("tg.test_render", icon="RENDER_STILL")
        col.operator("tg.export", icon="EXPORT")


def _draw_material_settings(layout, ms):
    """Draw a material's OpenRCT2 region/flags/shading settings inline."""
    layout.prop(ms, "region")
    col = layout.column(align=True)
    col.prop(ms, "is_mask")
    col.prop(ms, "is_visible_mask")
    col.prop(ms, "no_ao")
    col.prop(ms, "edge")
    col.prop(ms, "dark_edge")
    col.prop(ms, "no_bleed")
    col.prop(ms, "flat_shaded")
    layout.prop(ms, "texture")
    draw_bake(layout.column(align=True), ms)

    col = layout.column(align=True)
    col.label(text="Shading")
    row = col.row(align=True)
    row.prop(ms, "use_color_override", text="")
    sub = row.row()
    sub.enabled = ms.use_color_override
    sub.prop(ms, "diffuse_color", text="Color")
    col.prop(ms, "specular_exponent")
    col.prop(ms, "specular_intensity")
    row = col.row(align=True)
    row.prop(ms, "use_specular_tint", text="")
    sub = row.row()
    sub.enabled = ms.use_specular_tint
    sub.prop(ms, "specular_tint", text="Specular Tint")


def _draw_object_settings(layout, obj):
    """Draw the active object's track role + its materials."""
    layout.prop(obj.tg_object, "role")
    role = obj.tg_object.role
    if role == "IGNORE":
        return
    if role == "SPECIAL":
        layout.prop(obj.tg_object, "special_model")

    box = layout.box()
    box.label(text="Materials", icon="MATERIAL")
    if not obj.material_slots:
        box.label(text="No materials on this object.", icon="INFO")
        return
    if len(obj.material_slots) > 1:
        box.template_list(
            "MATERIAL_UL_matslots",
            "",
            obj,
            "material_slots",
            obj,
            "active_material_index",
            rows=2,
        )
    mat = obj.active_material
    if mat is None:
        box.label(text="Empty material slot.", icon="INFO")
    else:
        _draw_material_settings(box, mat.tg_material)


# --- Shared "Selected Object" container -------------------------------------
# Registered cooperatively with the vehicle/scenery add-ons (guarded by idname)
# so a single "Selected Object" panel hosts whichever add-ons are installed. The
# copies MUST keep the same bl_idname.
_SHARED_PARENT_IDNAME = "OPENRCT2_PT_selected_object"


class OPENRCT2_PT_selected_object(Panel):
    bl_idname = _SHARED_PARENT_IDNAME
    bl_label = "Selected Object"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRCT2"
    bl_order = 1

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == "MESH"

    def draw(self, context):
        pass


def _register_shared_parent():
    if not hasattr(bpy.types, _SHARED_PARENT_IDNAME):
        bpy.utils.register_class(OPENRCT2_PT_selected_object)


def _unregister_shared_parent():
    cls = getattr(bpy.types, _SHARED_PARENT_IDNAME, None)
    if cls is None:
        return
    for name in dir(bpy.types):
        if getattr(getattr(bpy.types, name, None), "bl_parent_id", "") == _SHARED_PARENT_IDNAME:
            return
    bpy.utils.unregister_class(cls)


class TG_PT_object_view3d(Panel):
    """The active object's track settings, as a child of "Selected Object"."""

    bl_label = "Track"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRCT2"
    bl_parent_id = _SHARED_PARENT_IDNAME
    bl_order = 2

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == "MESH" and hasattr(obj, "tg_object")

    def draw(self, context):
        _draw_object_settings(self.layout, context.object)


_CLASSES = (
    TG_UL_sections,
    TG_UL_lights,
    TG_PT_track,
    TG_PT_object_view3d,
)


def register():
    _register_shared_parent()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    _unregister_shared_parent()
