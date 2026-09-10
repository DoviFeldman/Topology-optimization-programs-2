#!/usr/bin/env python
"""
Independent FE check of a finished run: how stiff is the optimized part,
really, compared with the original solid one?

The optimizer's own compliance number is computed on a *grey* density field
mid-optimization. This re-solves from scratch on the **thresholded 0/1 design**
that was actually exported to STL, under the same load cases, and against the
untouched solid part as a reference. No optimization, just one FE solve each.

    python runners/validate.py f1_hero f3_chunky f4_feet
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engines import mesh_io, simp_fast as sf
from runners.run_simp import LOAD_PRESETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# every design is additionally scored under this one load case, so runs that
# used different presets can be put on the same axis
REFERENCE_PRESET = "squash_plus_sway_both"


def compliance_of(domain, supports, load_cases, weights, nu=0.3):
    """
    One FE solve per load case on a fully solid `domain`.

    volfrac=1, penal=1 and rmin=1.0 (a filter whose only non-zero stencil entry
    is the element itself) make the density field exactly 1 everywhere inside
    the domain, so iteration 1's compliance is a plain linear-elastic solve of
    that shape — no optimization involved.
    """
    res = sf.optimize(domain=domain, volfrac=1.0, penal=1.0, rmin=1.0,
                      max_iter=1, supports=supports, load_cases=load_cases,
                      case_weights=weights, projection=False,
                      penal_continuation=False, verbose=False, nu=nu)
    # `volume` in the history is measured after that iteration's OC update;
    # `vol_analyzed` is the field the compliance was actually computed on.
    analyzed = res["hist"][0]["vol_analyzed"]
    assert abs(analyzed - 1.0) < 1e-9, (
        "reference solve was not fully solid (density mean %.6f)" % analyzed)
    return res["compliance"]


def validate(run_id):
    rec = json.load(open(os.path.join(ROOT, "runs", run_id, "record.json")))
    cfg = rec["config"]
    stl = cfg["stl"] if os.path.isabs(cfg["stl"]) else os.path.join(ROOT, cfg["stl"])
    domain, meta = mesh_io.stl_to_domain(stl, resolution=cfg["resolution"],
                                         up_axis=cfg.get("up_axis", "z"))
    rho = np.load(os.path.join(ROOT, "runs", run_id, "density.npy"))
    solid = rho >= cfg.get("threshold", 0.5)

    preset = LOAD_PRESETS[cfg["preset"]]
    sup, lcs = preset["supports"], preset["load_cases"]
    w = preset.get("case_weights")

    # (a) under the load case the design was optimized for.
    # If the thresholded design has eroded away from the loaded or supported
    # face, there is no load path at all — that is a real result about the part,
    # not a harness failure, so record it rather than raising.
    try:
        c_opt = compliance_of(solid, sup, lcs, w)
    except ValueError as exc:
        out = dict(run=run_id, unloadable=True, reason=str(exc),
                   volume_optimized_mm3=float(solid.sum()) * meta["pitch"] ** 3,
                   volume_solid_mm3=float(domain.sum()) * meta["pitch"] ** 3)
        out["volume_ratio"] = out["volume_optimized_mm3"] / out["volume_solid_mm3"]
        rec["validation"] = out
        with open(os.path.join(ROOT, "runs", run_id, "record.json"), "w") as f:
            json.dump(rec, f, indent=2)
        return out
    c_solid = compliance_of(domain, sup, lcs, w)

    # A compliance that is zero or negative means the FE system was singular:
    # the exported part is a fragment that is not both loaded and supported.
    # Report that rather than a meaningless ratio.
    if c_opt <= 0 or not np.isfinite(c_opt):
        out = dict(run=run_id, unloadable=True,
                   reason="the exported part gives a singular FE system "
                          "(compliance %.3e) — it is a disconnected fragment "
                          "that is not both loaded and supported" % c_opt,
                   volume_optimized_mm3=float(solid.sum()) * meta["pitch"] ** 3,
                   volume_solid_mm3=float(domain.sum()) * meta["pitch"] ** 3)
        out["volume_ratio"] = out["volume_optimized_mm3"] / out["volume_solid_mm3"]
        rec["validation"] = out
        with open(os.path.join(ROOT, "runs", run_id, "record.json"), "w") as f:
            json.dump(rec, f, indent=2)
        return out

    # (b) under one COMMON reference load, so runs that used different presets
    #     can be compared with each other. A design optimized for corner feet is
    #     being judged here on a problem it was not asked to solve, so read this
    #     column as "how general is this shape", not as "who won".
    ref = LOAD_PRESETS[REFERENCE_PRESET]
    c_opt_ref = compliance_of(solid, ref["supports"], ref["load_cases"],
                              ref.get("case_weights"))
    c_solid_ref = compliance_of(domain, ref["supports"], ref["load_cases"],
                                ref.get("case_weights"))

    v_opt = float(solid.sum()) * meta["pitch"] ** 3
    v_solid = float(domain.sum()) * meta["pitch"] ** 3
    out = dict(
        run=run_id,
        compliance_optimized=c_opt, compliance_original_solid=c_solid,
        stiffness_ratio=c_solid / c_opt,          # <1 means softer than solid
        reference_preset=REFERENCE_PRESET,
        compliance_optimized_ref=c_opt_ref,
        compliance_original_solid_ref=c_solid_ref,
        stiffness_ratio_ref=c_solid_ref / c_opt_ref,
        volume_optimized_mm3=v_opt, volume_solid_mm3=v_solid,
        volume_ratio=v_opt / v_solid,
        stiffness_per_mass=(c_solid / c_opt) / (v_opt / v_solid),
        stiffness_per_mass_ref=(c_solid_ref / c_opt_ref) / (v_opt / v_solid),
    )
    rec["validation"] = out
    with open(os.path.join(ROOT, "runs", run_id, "record.json"), "w") as f:
        json.dump(rec, f, indent=2)
    return out


if __name__ == "__main__":
    ids = sys.argv[1:] or [d for d in sorted(os.listdir(os.path.join(ROOT, "runs")))
                           if d.startswith("f")]
    print("%-18s %8s %8s %10s | %8s %10s" %
          ("run", "mass%", "stiff%", "stiff/mass", "stiff%ref", "s/m ref"))
    for rid in ids:
        try:
            o = validate(rid)
        except Exception as exc:
            print("%-16s FAILED: %s" % (rid, exc))
            continue
        if o.get("unloadable"):
            print("%-18s %7.1f%%   UNLOADABLE: %s"
                  % (rid, 100 * o["volume_ratio"], o["reason"][:70]))
            continue
        print("%-18s %7.1f%% %7.1f%% %10.2f | %7.1f%% %10.2f" %
              (rid, 100 * o["volume_ratio"], 100 * o["stiffness_ratio"],
               o["stiffness_per_mass"], 100 * o["stiffness_ratio_ref"],
               o["stiffness_per_mass_ref"]))
