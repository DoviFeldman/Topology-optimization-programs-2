"""
Topology Optimization Studio — local Mac build.

    bash run.sh          # then open http://localhost:8501

Upload an STL (or use one that ships with this repo), point the load at it,
choose an engine, and run. The optimized result previews in 3-D and downloads
as a printable STL.

Three engines are wired in and all three run natively on Apple Silicon:

  1. 3D SIMP  — this repo's own engine, rewritten for speed (AMG-preconditioned
                CG, vectorized filter, multiple load cases, Heaviside projection).
  2. PyTopo3D — the upstream package (jihoonkim888/PyTopo3D), driven through its
                obstacle/support/force-field API.
  3. ToPy     — upstream williamhunter/ToPy, ported here to Python 3 with a
                SciPy-backed stand-in for PySparse.
"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import streamlit as st

from engines import mesh_io, simp_fast as sf
from runners.run_simp import LOAD_PRESETS, build_keep

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(ROOT, "uploads")
RUNS = os.path.join(ROOT, "runs")
os.makedirs(UPLOADS, exist_ok=True)

st.set_page_config(page_title="Topology Optimization Studio", layout="wide",
                   page_icon="🕸️")

FACES = ["z+", "z-", "x+", "x-", "y+", "y-"]
PATTERNS = ["all", "center", "ring", "corners", "strip_u", "strip_v",
            "edge_u0", "edge_u1", "edge_v0", "edge_v1", "half_u", "half_v"]
DIRS = ["-z", "+z", "-x", "+x", "-y", "+y"]


# ------------------------------------------------------------------ helpers
def mesh_figure(mesh, color="#6fa8dc", title=""):
    import plotly.graph_objects as go
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    fig = go.Figure(go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.45, diffuse=0.9, specular=0.25, roughness=0.6),
        lightposition=dict(x=200, y=-200, z=400)))
    lo, hi = v.min(0), v.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    fig.update_layout(
        title=title, height=620, margin=dict(l=0, r=0, t=34, b=0),
        scene=dict(aspectmode="cube",
                   xaxis=dict(range=[c[0] - r, c[0] + r], title="X (mm)"),
                   yaxis=dict(range=[c[1] - r, c[1] + r], title="Y (mm)"),
                   zaxis=dict(range=[c[2] - r, c[2] + r], title="Z (mm) — up"),
                   camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0))))
    return fig


def bc_editor(label, kind, key):
    """Let the user build their own support / load list."""
    rows = []
    n = st.number_input("How many %s groups?" % label, 1, 4, 1, key=key + "_n")
    for i in range(int(n)):
        cols = st.columns(4 if kind == "load" else 4)
        face = cols[0].selectbox("face", FACES, key="%s_f%d" % (key, i))
        pat = cols[1].selectbox("region", PATTERNS, key="%s_p%d" % (key, i))
        frac = cols[2].slider("size", 0.05, 1.0, 0.3, 0.05, key="%s_r%d" % (key, i))
        if kind == "load":
            d = cols[3].selectbox("direction", DIRS, key="%s_d%d" % (key, i))
            rows.append({"face": face, "pattern": pat, "frac": frac, "dir": d})
        else:
            dofs = cols[3].multiselect("held axes", ["x", "y", "z"],
                                       default=["x", "y", "z"],
                                       key="%s_x%d" % (key, i))
            rows.append({"face": face, "pattern": pat, "frac": frac,
                         "dofs": "".join(dofs) or "xyz"})
    return rows


# --------------------------------------------------------------------- UI
st.title("🕸️ Topology Optimization Studio")

tab_run, tab_gallery, tab_about = st.tabs(
    ["Optimize", "Past runs", "Engines & how it works"])

with tab_run:
    left, right = st.columns([1, 1.5], gap="large")

    with left:
        st.subheader("1 · Geometry")
        up = st.file_uploader("Upload an STL", type=["stl"])
        bundled = sorted(f for f in os.listdir(ROOT) if f.lower().endswith(".stl"))
        prev_up = sorted(os.listdir(UPLOADS)) if os.path.isdir(UPLOADS) else []
        choices = ["(use the file I just uploaded)"] if up else []
        choices += bundled + ["uploads/" + f for f in prev_up]
        pick = st.selectbox("…or pick a file already here", choices) if choices else None

        if up is not None:
            stl_path = os.path.join(UPLOADS, up.name)
            with open(stl_path, "wb") as f:
                f.write(up.getbuffer())
            if pick and not pick.startswith("("):
                stl_path = os.path.join(ROOT, pick)
        elif pick:
            stl_path = os.path.join(ROOT, pick)
        else:
            stl_path = None

        up_axis = st.selectbox(
            "Which axis of the model points UP?", ["z", "y", "x"],
            help="The load presets push down along the grid's +Z. If your STL "
                 "stands up along Y, choose Y and it is rotated for you.")
        resolution = st.slider(
            "Resolution (voxels along the longest axis)", 24, 200, 80, 4,
            help="Cost grows roughly with the cube of this. 64–96 explores "
                 "quickly; 120–160 is for the final run.")

        if stl_path and os.path.exists(stl_path):
            try:
                dom, meta = mesh_io.stl_to_domain(stl_path, resolution, up_axis)
                st.caption(
                    "Grid %d × %d × %d · %s voxels of material · voxel = %.2f mm · "
                    "part is %.0f × %.0f × %.0f mm"
                    % (*dom.shape, format(int(dom.sum()), ","), meta["pitch"],
                       *meta["extents"]))
                st.session_state["_dom_shape"] = dom.shape
            except Exception as exc:
                st.error("Could not voxelize that STL: %s" % exc)
                dom = None
        else:
            dom = None
            st.info("Upload an STL to begin.")

        st.subheader("2 · Load case")
        st.caption("Everything below is 'the force comes from the top' unless "
                   "you build your own.")
        mode = st.radio("Boundary conditions", ["Preset", "Build my own"],
                        horizontal=True)
        if mode == "Preset":
            preset_name = st.selectbox("Preset", list(LOAD_PRESETS),
                                       index=list(LOAD_PRESETS).index(
                                           "squash_plus_sway_both"))
            preset = LOAD_PRESETS[preset_name]
            st.info(preset["blurb"])
            supports = preset["supports"]
            load_cases = preset["load_cases"]
            case_weights = preset.get("case_weights")
        else:
            preset_name = "custom"
            st.markdown("**Supports** (where the part is held)")
            supports = bc_editor("support", "support", "sup")
            st.markdown("**Loads** (where the force is applied)")
            loads = bc_editor("load", "load", "ld")
            multi = st.checkbox(
                "Treat each load group as a separate load case", value=True,
                help="Separate cases means the part must survive each one on "
                     "its own. That is what breaks the 'solid column' answer "
                     "you get from a single, purely vertical squash.")
            if multi:
                load_cases = [[ld] for ld in loads]
                case_weights = [
                    st.slider("weight of load %d" % (i + 1), 0.05, 2.0,
                              1.0 if i == 0 else 0.3, 0.05, key="w%d" % i)
                    for i in range(len(loads))]
            else:
                load_cases = [loads]
                case_weights = None

        st.subheader("3 · Optimizer")
        engine = st.selectbox(
            "Engine", ["3D SIMP (built in, fastest)", "PyTopo3D (upstream)",
                       "ToPy (upstream, ported)"])
        c1, c2 = st.columns(2)
        volfrac = c1.slider("Volume fraction (how much material to keep)",
                            0.05, 0.9, 0.30, 0.01)
        penal = c2.slider("Penalization p (pushes densities to 0 or 1)",
                          1.0, 6.0, 3.0, 0.25)
        rmin = c1.slider("Filter radius (min feature size, in voxels)",
                         1.1, 6.0, 2.2, 0.1)
        max_iter = c2.slider("Iterations", 10, 150, 55, 5)

        with st.expander("Advanced"):
            projection = st.checkbox(
                "Heaviside projection (crisper struts)", value=True,
                help="Only for the built-in engine.")
            beta_max = st.slider("Projection strength β max", 2.0, 64.0, 16.0, 2.0)
            threshold = st.slider("Density threshold for the STL", 0.2, 0.8, 0.5, 0.05)
            smooth_iters = st.slider("Mesh smoothing passes", 0, 30, 8, 1)
            keep_largest = st.checkbox("Drop disconnected floating bits", True)
            eta = st.select_slider("ToPy: eta (damping)",
                                   ["0.1", "0.2", "0.3", "0.4", "0.5", "exp"],
                                   value="0.4")
            q_max = st.slider("ToPy: grey-scale filter q_max", 1.0, 4.0, 2.0, 0.25)
            st.markdown("**Keep regions** — forced solid, so the part stays usable")
            keep_base = st.slider("keep bottom (fraction of height)", 0.0, 0.4, 0.0, 0.02)
            keep_top = st.slider("keep top (fraction of height)", 0.0, 0.4, 0.0, 0.02)

        run_name = st.text_input("Name this run", "my_run_%s" % time.strftime("%H%M%S"))
        go = st.button("Run optimization", type="primary",
                       disabled=stl_path is None)

    with right:
        if go and stl_path:
            keep_specs = []
            if keep_base > 0:
                keep_specs.append({"type": "box", "z": [0.0, keep_base]})
            if keep_top > 0:
                keep_specs.append({"type": "box", "z": [1.0 - keep_top, 1.0]})

            cfg = dict(id=run_name, stl=stl_path, resolution=resolution,
                       up_axis=up_axis, volfrac=volfrac, penal=penal, rmin=rmin,
                       max_iter=max_iter, preset=preset_name,
                       projection=projection, beta_max=beta_max,
                       threshold=threshold, smooth_iters=smooth_iters,
                       keep_largest=keep_largest, keep=keep_specs or None,
                       eta=eta, q_max=q_max, verbose=False)

            prog = st.progress(0.0, "starting…")
            chart = st.empty()
            hist = []

            if engine.startswith("3D SIMP"):
                domain, meta = mesh_io.stl_to_domain(stl_path, resolution, up_axis)
                keep = build_keep(domain, keep_specs) if keep_specs else None

                def cb(h, xPhys, idx):
                    hist.append(h)
                    prog.progress(min(h["it"] / max_iter, 1.0),
                                  "iteration %d — compliance %.4g, %.0f%% grey"
                                  % (h["it"], h["compliance"], 100 * h["grayness"]))
                    if h["it"] % 3 == 0:
                        chart.line_chart(
                            {"compliance": [x["compliance"] for x in hist]})

                t0 = time.time()
                res = sf.optimize(
                    domain=domain, keep=keep, volfrac=volfrac, penal=penal,
                    rmin=rmin, max_iter=max_iter, supports=supports,
                    load_cases=load_cases, case_weights=case_weights,
                    projection=projection, beta_max=beta_max,
                    callback=cb, verbose=False)
                rho, secs = res["rho"], res["seconds"]
                extra = "compliance %.4e · %d elements · %d dofs" % (
                    res["compliance"], res["na"], res["nfree"])
            else:
                import subprocess
                import sys as _sys
                runner = ("runners/run_pytopo3d.py"
                          if engine.startswith("PyTopo3D") else "runners/run_topy.py")
                cfgp = os.path.join(RUNS, "_cfg")
                os.makedirs(cfgp, exist_ok=True)
                cfgf = os.path.join(cfgp, run_name + ".json")
                cfg["engine"] = "pytopo3d" if engine.startswith("PyTopo3D") else "topy"
                with open(cfgf, "w") as f:
                    json.dump(cfg, f)
                prog.progress(0.05, "running %s in a subprocess…" % engine)
                t0 = time.time()
                p = subprocess.run(
                    [os.path.join(ROOT, ".venv", "bin", "python"),
                     os.path.join(ROOT, runner), cfgf],
                    cwd=ROOT, capture_output=True, text=True)
                if p.returncode != 0:
                    st.error("That engine failed:")
                    st.code(p.stdout[-4000:] + "\n" + p.stderr[-4000:])
                    st.stop()
                rec = json.load(open(os.path.join(RUNS, run_name, "record.json")))
                rho = np.load(os.path.join(RUNS, run_name, "density.npy"))
                _d, meta = mesh_io.stl_to_domain(stl_path, resolution, up_axis)
                secs = rec["wall_seconds"]
                extra = "volume fraction %.3f" % rec["volume_fraction"]

            prog.progress(1.0, "meshing…")
            mesh, minfo = mesh_io.density_to_mesh(
                rho, threshold=threshold, pitch=meta["pitch"],
                smooth_iters=smooth_iters, keep_largest=keep_largest)
            outdir = os.path.join(RUNS, run_name)
            os.makedirs(outdir, exist_ok=True)
            out_stl = os.path.join(outdir, run_name + ".stl")
            mesh_io.save_stl(mesh, out_stl)
            np.save(os.path.join(outdir, "density.npy"), rho.astype(np.float32))
            with open(os.path.join(outdir, "settings.json"), "w") as f:
                json.dump(dict(cfg, supports=supports, load_cases=load_cases,
                               case_weights=case_weights, engine=engine),
                          f, indent=2, default=str)

            prog.empty()
            st.success("Done in %.0f s — %s" % (secs, extra))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Part volume", "%.0f mm³" % minfo["volume_mm3"])
            if meta["mesh_volume_mm3"]:
                m2.metric("Mass saved",
                          "%.0f %%" % (100 * (1 - minfo["volume_mm3"] /
                                              meta["mesh_volume_mm3"])))
            m3.metric("Triangles", format(minfo["faces"], ","))
            m4.metric("Watertight", "yes" if minfo["watertight"] else "no")
            if minfo.get("components", 1) > 1:
                st.warning("The result came out in %d pieces; %d kept."
                           % (minfo["components"], minfo["components_kept"]))
            st.plotly_chart(mesh_figure(mesh, title=run_name),
                            use_container_width=True)
            with open(out_stl, "rb") as f:
                st.download_button("⬇ Download STL", f, file_name=run_name + ".stl",
                                   mime="model/stl", type="primary")
        else:
            st.markdown(
                "### How to get the good-looking result\n"
                "* A pillar squashed **straight down**, held across its whole "
                "base, has *vertical columns* as its exact optimum — you get a "
                "slab, not a lattice. That is correct physics, not a bug.\n"
                "* To get organic bracing, give the structure something to "
                "resist sideways: use a **multi-load-case** preset "
                "(`squash_plus_sway_both`), or stand the part on **corner "
                "feet** so the load has to travel diagonally.\n"
                "* Lower **volume fraction** → thinner, more skeletal members.\n"
                "* Larger **filter radius** → fewer, chunkier, more organic "
                "members. Smaller → fine lace (and more print risk).\n"
                "* **Resolution** is what buys detail. 64–96 to explore, "
                "120–160 for the final.")
            if stl_path and os.path.exists(stl_path):
                import trimesh
                m = trimesh.load(stl_path, force="mesh")
                st.plotly_chart(mesh_figure(m, "#9aa5b1", "input geometry"),
                                use_container_width=True)

with tab_gallery:
    st.subheader("Every run in `runs/`")
    recs = []
    for d in sorted(os.listdir(RUNS)):
        rp = os.path.join(RUNS, d, "record.json")
        if os.path.exists(rp):
            try:
                recs.append(json.load(open(rp)))
            except Exception:
                pass
    if not recs:
        st.info("No completed batch runs yet.")
    for r in recs:
        with st.expander("%s  —  %s" % (r["config"]["id"],
                                        r.get("preset_blurb", ""))):
            c = r["config"]
            st.write({k: c[k] for k in
                      ("engine", "resolution", "volfrac", "penal", "rmin",
                       "max_iter", "preset") if k in c})
            png = os.path.join(ROOT, r["preview"])
            if os.path.exists(png):
                st.image(png)
            stl = os.path.join(ROOT, r["stl"])
            if os.path.exists(stl):
                with open(stl, "rb") as f:
                    st.download_button("⬇ STL", f, key="dl" + c["id"],
                                       file_name=os.path.basename(stl))

with tab_about:
    st.markdown(__doc__)
    st.markdown(
        "### What each setting does\n"
        "| Setting | Effect |\n|---|---|\n"
        "| **Volume fraction** | Share of the design domain kept as material. "
        "Lower = lighter and more skeletal. |\n"
        "| **Penalization p** | How hard intermediate densities are punished. "
        "p=1 gives grey mush; p=3 is standard; p≥4 gives crisp but more "
        "local-minimum-prone results. |\n"
        "| **Filter radius** | Minimum feature size in voxels. Sets member "
        "thickness and stops checkerboarding. |\n"
        "| **Resolution** | Voxels along the longest axis. The single biggest "
        "lever on how detailed and organic the result looks. |\n"
        "| **Projection β** | Heaviside sharpening. Higher = closer to true "
        "0/1, so the printed part matches what the solver actually analyzed. |\n"
        "| **ToPy eta / q** | ToPy's own damping and grey-scale filter — its "
        "equivalent of projection. |\n"
        "| **Keep regions** | Elements forced solid so mounting faces survive "
        "the optimization. |\n")
    if os.path.exists(os.path.join(ROOT, "REPORT.md")):
        st.markdown("---")
        st.markdown(open(os.path.join(ROOT, "REPORT.md")).read())
