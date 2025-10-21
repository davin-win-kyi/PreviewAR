from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional, Tuple

# ---------- Units ----------
_UNIT_TO_M = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254, '"': 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
}

def _to_meters(value: float, unit: Optional[str], default_unit: str = "m") -> float:
    u = (unit or default_unit).strip().lower()
    if u not in _UNIT_TO_M:
        raise ValueError(f"Unknown unit '{u}'. Allowed: {sorted(_UNIT_TO_M)}")
    return float(value) * _UNIT_TO_M[u]

# ---------- 1) GPT dimensions extractor ----------
def get_dimensions_via_gpt(
    root_json: Dict[str, Any],
    *,
    model: str = "gpt-5",
    default_unit: str = "m"
) -> Dict[str, Any]:
    """
    Input:
      root_json: dict that contains `product` -> `specifications` (or `specification`).
    Returns:
      {
        "length_m": float,
        "width_m": float,
        "height_m": float,
        "gpt_raw": {
          "length": {"value": float, "unit": str}, "width": {...}, "height": {...}
        }
      }
    Requires: OPENAI_API_KEY in env.
    """
    # Pull the specification payload from the provided JSON (be forgiving about the key)
    product = root_json.get("product", root_json)
    spec_payload = (
        product.get("specifications")
        or product.get("specification")
        or product.get("specs")
        or product
    )

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI client not installed. Run: pip install openai") from e

    client = OpenAI()

    schema = {
        "type": "object",
        "properties": {
            "length": {"type": "object", "properties": {
                "value": {"type": "number"}, "unit": {"type": "string"}
            }, "required": ["value", "unit"]},
            "width": {"type": "object", "properties": {
                "value": {"type": "number"}, "unit": {"type": "string"}
            }, "required": ["value", "unit"]},
            "height": {"type": "object", "properties": {
                "value": {"type": "number"}, "unit": {"type": "string"}
            }, "required": ["value", "unit"]},
        },
        "required": ["length", "width", "height"],
        "additionalProperties": False,
    }

    sys_msg = (
        "Extract the product's external length, width, and height. "
        "Return ONLY JSON matching the schema with numeric values and canonical units "
        "(mm, cm, m, in, ft). If a single unit applies to all, apply it. "
        "If text uses L×W×H, map L→length, W→width, H→height."
    )

    user_blob = {
        "hint": "This is the `product -> specifications` field from a product JSON.",
        "specification_payload": spec_payload,
    }

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_schema", "json_schema": {"name": "dims", "schema": schema}},
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": json.dumps(user_blob, ensure_ascii=False)},
        ],
    )
    gpt_raw = json.loads(resp.choices[0].message.content)

    # Validate and convert to meters
    for k in ("length", "width", "height"):
        if k not in gpt_raw or "value" not in gpt_raw[k] or "unit" not in gpt_raw[k]:
            raise ValueError(f"GPT output missing {k}")

    length_m = _to_meters(gpt_raw["length"]["value"], gpt_raw["length"]["unit"], default_unit)
    width_m  = _to_meters(gpt_raw["width"]["value"],  gpt_raw["width"]["unit"],  default_unit)
    height_m = _to_meters(gpt_raw["height"]["value"], gpt_raw["height"]["unit"], default_unit)

    return {
        "length_m": length_m,
        "width_m": width_m,
        "height_m": height_m,
        "gpt_raw": gpt_raw,
    }

# ---------- 2) Blender scaler ----------
def scale_glb_in_blender(
    glb_path: str,
    length_m: float,
    width_m: float,
    height_m: float,
    out_path: Optional[str] = None
) -> str:
    """
    Scale a GLB so its world-space bounding box equals (length_m, width_m, height_m).
    Returns: output GLB path.
    Must be executed inside Blender's Python (bpy available).
    """
    try:
        import bpy
        import mathutils
    except Exception as e:
        raise RuntimeError("This function must run inside Blender (bpy not available).") from e

    glb_path = os.path.abspath(glb_path)
    out_path = os.path.abspath(out_path or (os.path.splitext(glb_path)[0] + "_scaled.glb"))

    # Fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=glb_path, loglevel=50)
    imported = [obj for obj in bpy.context.selected_objects]
    if not imported:
        imported = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not imported:
        raise RuntimeError("No mesh objects imported from GLB.")

    # Compute world bbox size
    def bbox_size_world(objs):
        mins = mathutils.Vector(( float("inf"),  float("inf"),  float("inf")))
        maxs = mathutils.Vector((-float("inf"), -float("inf"), -float("inf")))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in objs:
            ev = obj.evaluated_get(depsgraph)
            mesh = ev.to_mesh()
            if not mesh:
                continue
            for v in mesh.vertices:
                wv = ev.matrix_world @ v.co
                mins.x = min(mins.x, wv.x); mins.y = min(mins.y, wv.y); mins.z = min(mins.z, wv.z)
                maxs.x = max(maxs.x, wv.x); maxs.y = max(maxs.y, wv.y); maxs.z = max(maxs.z, wv.z)
            ev.to_mesh_clear()
        size = maxs - mins
        return (size.x, size.y, size.z)

    cur_x, cur_y, cur_z = bbox_size_world(imported)

    def safe_div(t, c):  # avoid div by ~0
        return 1.0 if c <= 1e-9 or t <= 1e-9 else (t / c)

    sx = safe_div(length_m, cur_x)
    sy = safe_div(width_m,  cur_y)
    sz = safe_div(height_m, cur_z)

    # Parent and scale per-axis
    parent = bpy.data.objects.new("ScaleParent", None)
    bpy.context.collection.objects.link(parent)
    for o in imported:
        o.parent = parent

    parent.scale[0] *= sx
    parent.scale[1] *= sy
    parent.scale[2] *= sz

    # Export
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format='GLB',
        use_selection=True,
        export_apply=False,
    )
    return out_path
