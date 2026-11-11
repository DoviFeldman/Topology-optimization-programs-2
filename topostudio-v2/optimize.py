#!/usr/bin/env python
"""
Command line:  STL in -> optimized STL out.

    python optimize.py part.stl out.stl --style organic --load stand
    python optimize.py part.stl out.stl --style skeleton --load cantilever --detail fine
    python optimize.py --list
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from engines import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--style", default="organic", choices=list(pipeline.STYLES))
    ap.add_argument("--load", default="stand", choices=list(pipeline.LOADS))
    ap.add_argument("--detail", default="standard", choices=list(pipeline.DETAIL))
    ap.add_argument("--up", default="z", choices=["x", "y", "z"],
                    help="which axis of the STL points up")
    ap.add_argument("--volfrac", type=float, help="override: fraction of material kept")
    ap.add_argument("--member", type=float,
                    help="override: member size as a fraction of the longest side")
    ap.add_argument("--list", action="store_true", help="list styles and load setups")
    a = ap.parse_args()
    if a.list or not a.input:
        print("STYLES")
        for k, v in pipeline.STYLES.items():
            print("  %-10s %s" % (k, v["blurb"]))
        print("LOAD SETUPS")
        for k, v in pipeline.LOADS.items():
            print("  %-12s %s" % (k, v["blurb"]))
        return

    def prog(h):
        print("  %-6s it %3d  compliance %.4g  change %.4f  grey %2.0f%%  %4.0fs"
              % (h["level"], h["it"], h["compliance"], h["change"],
                 100 * h["grayness"], h["seconds"]), flush=True)

    mesh, info, _, _ = pipeline.run(a.input, up_axis=a.up, style=a.style, load=a.load,
                                    detail=a.detail, volfrac=a.volfrac,
                                    member=a.member, progress=prog)
    out = a.output or os.path.splitext(a.input)[0] + "_optimized.stl"
    mesh.export(out)
    m = info["mesh"]
    print("wrote %s  |  %.0f s  |  %d triangles  |  watertight: %s  |  keeps %.0f%% "
          "of the material  |  outer faces exact: %s"
          % (out, info["seconds"], m["faces"], m["watertight"],
             100 * (info["mass_kept"] or 0), m["exact_outer"]))


if __name__ == "__main__":
    main()
