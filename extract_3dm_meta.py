#!/usr/bin/env python3
"""
Extract key measurements from a .3dm jewellery CAD file.

Usage:
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 extract_3dm_meta.py DPBN0002/DPBN0002.3dm

Requires: pip install rhino3dm  (python 3.13 pip)
"""
import sys, math
import rhino3dm

GEM_IDEF_NAMES = {"diamond", "gem", "stone", "pear", "emerald", "round", "oval", "cushion", "marquise"}
METAL_LAYERS   = {"Metal 01", "Metal 02", "Metal 03", "Metal 04"}
GEM_LAYERS     = {"Gem 01", "Gem 02", "Gem 03", "Gem 04", "Gem"}


def classify_cut(name: str) -> str:
    name = name.lower()
    for cut in ("pear", "emerald", "round", "oval", "cushion", "marquise", "princess", "radiant"):
        if cut in name:
            return cut
    return "unknown"


def bbox_dims(bb) -> tuple:
    """Return (dx, dy, dz) sorted as (long, med, depth)."""
    dx = abs(bb.Max.X - bb.Min.X)
    dy = abs(bb.Max.Y - bb.Min.Y)
    dz = abs(bb.Max.Z - bb.Min.Z)
    face = sorted([dx, dy], reverse=True)
    return face[0], face[1], dz


def extract(path: str) -> dict:
    f = rhino3dm.File3dm.Read(path)
    if not f:
        raise FileNotFoundError(f"Cannot read: {path}")

    units = str(f.Settings.ModelUnitSystem)  # e.g. "UnitSystem.Millimeters"
    unit  = "mm" if "Millimeter" in units else units

    # --- Build idef id → name map -----------------------------------------
    idef_map = {str(idef.Id): idef.Name for idef in f.InstanceDefinitions}

    # --- Walk objects -------------------------------------------------------
    gems_placed   = []   # InstanceReferences on gem layers
    bangle_curves = []   # large NurbsCurves / rings

    for obj in f.Objects:
        g = obj.Geometry
        a = obj.Attributes
        layer_name = f.Layers[a.LayerIndex].Name if a.LayerIndex < len(f.Layers) else ""
        try:
            bb = g.GetBoundingBox()
        except TypeError:
            continue
        if not bb:
            continue

        long_dim, med_dim, depth = bbox_dims(bb)

        # Placed gem instances
        if isinstance(g, rhino3dm.InstanceReference) and layer_name in GEM_LAYERS:
            idef_name = idef_map.get(str(g.ParentIdefId), "?")
            cut = classify_cut(idef_name)
            gems_placed.append({
                "idef":  idef_name,
                "cut":   cut,
                "long":  round(long_dim, 3),
                "width": round(med_dim, 3),
                "depth": round(depth, 3),
                "layer": layer_name,
            })

        # Bangle ring path (large open curve)
        if isinstance(g, rhino3dm.NurbsCurve) and long_dim > 30:
            bangle_curves.append(round(long_dim, 2))

    # --- Deduplicate gems by cut type (keep largest) -----------------------
    by_cut: dict[str, dict] = {}
    for gem in gems_placed:
        c = gem["cut"]
        if c not in by_cut or gem["long"] > by_cut[c]["long"]:
            by_cut[c] = gem

    # --- Bangle diameter ----------------------------------------------------
    bangle_diam = max(bangle_curves) if bangle_curves else None

    return {
        "source":        path,
        "unit":          unit,
        "bangle_diam_mm": bangle_diam,
        "gems":          list(by_cut.values()),
        "all_gem_instances": gems_placed,
    }


def pretty_print(meta: dict):
    print("=" * 60)
    print(f"FILE  : {meta['source']}")
    print(f"UNITS : {meta['unit']}")
    print("=" * 60)

    if meta["bangle_diam_mm"]:
        print(f"\nBANGLE outer diameter : {meta['bangle_diam_mm']} mm")

    print(f"\nGEM STONES (unique cuts, actual placed dimensions):")
    for g in meta["gems"]:
        print(f"  {g['cut'].upper():10s}  "
              f"length={g['long']:.2f}mm  width={g['width']:.2f}mm  depth={g['depth']:.2f}mm"
              f"  [{g['idef']}]")

    print(f"\nALL PLACED INSTANCES:")
    for g in meta["all_gem_instances"]:
        print(f"  {g['cut']:10s}  {g['long']:.3f} x {g['width']:.3f} x {g['depth']:.3f} mm"
              f"  layer={g['layer']}")

    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "DPBN0002/DPBN0002.3dm"
    meta = extract(path)
    pretty_print(meta)
