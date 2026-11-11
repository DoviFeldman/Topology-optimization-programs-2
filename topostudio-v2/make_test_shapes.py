#!/usr/bin/env python
"""Write a few test STLs into ./samples (Z is up, millimetres)."""
import os

import numpy as np
import trimesh
from trimesh.creation import box, cylinder

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def tbox(ext, lo):
    b = box(extents=ext)
    b.apply_translation(np.array(lo) + np.array(ext) / 2.0)
    return b


def union(parts):
    return trimesh.boolean.union(parts, engine="manifold")


def main():
    os.makedirs(OUT, exist_ok=True)
    tbox((60, 60, 90), (0, 0, 0)).export(os.path.join(OUT, "pillar.stl"))
    tbox((120, 40, 60), (0, 0, 0)).export(os.path.join(OUT, "beam.stl"))
    union([tbox((80, 40, 30), (0, 0, 0)), tbox((30, 40, 80), (0, 0, 0))]
          ).export(os.path.join(OUT, "lbracket.stl"))
    c = cylinder(radius=30, height=70, sections=96)
    c.apply_translation((30, 30, 35))
    bore = cylinder(radius=9, height=80, sections=64)
    bore.apply_translation((30, 30, 35))
    trimesh.boolean.difference([c, bore], engine="manifold").export(
        os.path.join(OUT, "cylinder.stl"))
    union([tbox((33, 3.7, 78), (0, 0, 0)), tbox((3.5, 24, 78), (0, 0, 0)),
           tbox((3.5, 24, 78), (29.5, 0, 0)), tbox((3.0, 24, 78), (15, 0, 0)),
           tbox((33, 24, 5), (0, 0, 0))]).export(os.path.join(OUT, "holder.stl"))
    for f in sorted(os.listdir(OUT)):
        m = trimesh.load(os.path.join(OUT, f))
        print("%-14s %s mm  watertight=%s" % (f, m.extents.round(1), m.is_watertight))


if __name__ == "__main__":
    main()
