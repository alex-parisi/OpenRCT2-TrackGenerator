"""Blender operators for the track add-on: test render, threaded export, and
section/light list management.

NOTE: no ``from __future__ import annotations``; operators declare bpy properties
as annotations and PEP 563 would break registration.
"""

import os
import tempfile
import time

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from openrct2_object_common.blender.lights import lights_from_items
from openrct2_object_common.blender.modal import RenderModalBase
from openrct2_object_common.cli import make_context
from openrct2_track_generator.exporter import export_track, export_track_test
from openrct2_track_generator.loader import build_track

from . import scene_to_track


class _TrackModalBase(RenderModalBase):
    """Shared base: build the Track on the main thread, render off-thread."""

    _clean_error_types = (scene_to_track.SceneError,)
    _invalid_prefix = "Invalid track"

    def _build(self, context):
        config, meshes, special_models, preview = scene_to_track.build_config_and_meshes(context)
        return build_track(config, meshes, preview, special_models)

    def _prepare(self, context, payload) -> None:
        self._lights = lights_from_items(context.scene.tg_track.lights)


class TG_OT_test_render(_TrackModalBase):
    bl_idname = "tg.test_render"
    bl_label = "Test Render"
    bl_description = "Render the track's sections and show the first in the Image Editor"

    _status_verb = "Rendering test"

    def _prepare(self, context, payload) -> None:
        super()._prepare(context, payload)
        self._tmp = tempfile.mkdtemp(prefix="tg_test_")
        self._png = None

    def _render(self, payload) -> None:
        track = payload
        ctx = make_context(self._lights, track.units_per_tile, True)
        export_track_test(track, ctx, self._tmp)
        if track.sections:
            self._png = os.path.join(self._tmp, f"{track.sections[0].name}.png")

    def _on_success(self, context):
        if not self._png or not os.path.exists(self._png):
            self.report({"WARNING"}, "Render produced no sprite")
            return {"CANCELLED"}
        img = bpy.data.images.load(self._png, check_existing=False)
        for area in context.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = img
                break
        self.report({"INFO"}, f"Test sprite loaded: {img.name}")
        return {"FINISHED"}


class TG_OT_export(_TrackModalBase):
    bl_idname = "tg.export"
    bl_label = "Export Sprites"
    bl_description = "Render every section and write the sprite manifest + indexed PNGs"

    _status_verb = "Exporting sprites"

    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def _prepare(self, context, payload) -> None:
        super()._prepare(context, payload)
        self._out = bpy.path.abspath(self.directory) if self.directory else os.getcwd()

    def _render(self, payload) -> None:
        track = payload
        ctx = make_context(self._lights, track.units_per_tile, False)
        export_track(track, ctx, self._out)

    def _on_success(self, context):
        elapsed = int(time.monotonic() - self._start_time)
        build = f" (build {self._build_secs}s)" if self._build_secs else ""
        self.report({"INFO"}, f"Exported track sprites to {self._out} in {elapsed}s{build}")
        return {"FINISHED"}


class TG_OT_section_add(Operator):
    bl_idname = "tg.section_add"
    bl_label = "Add Section"
    bl_description = "Add a section to generate"

    def execute(self, context):
        tt = context.scene.tg_track
        tt.sections.add()
        tt.section_index = len(tt.sections) - 1
        return {"FINISHED"}


class TG_OT_section_remove(Operator):
    bl_idname = "tg.section_remove"
    bl_label = "Remove Section"
    bl_description = "Remove the selected section"

    def execute(self, context):
        tt = context.scene.tg_track
        if not tt.sections:
            return {"CANCELLED"}
        tt.sections.remove(tt.section_index)
        tt.section_index = max(0, min(tt.section_index, len(tt.sections) - 1))
        return {"FINISHED"}


class TG_OT_light_add(Operator):
    bl_idname = "tg.light_add"
    bl_label = "Add Light"
    bl_description = "Add a light to the custom lighting rig"

    def execute(self, context):
        tt = context.scene.tg_track
        tt.lights.add()
        tt.light_index = len(tt.lights) - 1
        return {"FINISHED"}


class TG_OT_light_remove(Operator):
    bl_idname = "tg.light_remove"
    bl_label = "Remove Light"
    bl_description = "Remove the selected light"

    def execute(self, context):
        tt = context.scene.tg_track
        if not tt.lights:
            return {"CANCELLED"}
        tt.lights.remove(tt.light_index)
        tt.light_index = max(0, min(tt.light_index, len(tt.lights) - 1))
        return {"FINISHED"}


_CLASSES = (
    TG_OT_test_render,
    TG_OT_export,
    TG_OT_section_add,
    TG_OT_section_remove,
    TG_OT_light_add,
    TG_OT_light_remove,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
