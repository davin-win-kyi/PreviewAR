# import_glb.py
import sys, os
import bpy

# Get .glb path from CLI (after the first "--")
if "--" in sys.argv:
    glb_path = sys.argv[sys.argv.index("--") + 1]
else:
    raise SystemExit("Usage: blender --python import_glb.py -- /path/to/model.glb")

glb_path = os.path.abspath(os.path.expanduser(glb_path))
if not os.path.isfile(glb_path):
    raise SystemExit(f"File not found: {glb_path}")

# Clean empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Ensure glTF importer is available (usually already enabled)
try:
    bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
except Exception:
    pass

# Import the GLB
res = bpy.ops.import_scene.gltf(filepath=glb_path)
if 'CANCELLED' in res:
    raise RuntimeError(f"Failed to import {glb_path}")

# Optional niceties: frame view on imported content
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        with bpy.context.temp_override(area=area, region=area.regions[-1], window=bpy.context.window):
            bpy.ops.view3d.view_all(center=True)
        break

print("Imported objects:", [o.name for o in bpy.data.objects])
