"""OpenRCT2 Track Generator: Blender add-on entry point.

Registration order matters: PropertyGroups (and their Scene/Object/Material
pointer properties) must exist before the panels that draw them. The renderer
and the track maths live in the bundled `openrct2_track_generator` /
`openrct2_x7_renderer` wheels; this package is only the UI + scene adapter.
"""

from . import operators, panels, props


def register():
    props.register()
    operators.register()
    panels.register()


def unregister():
    panels.unregister()
    operators.unregister()
    props.unregister()
