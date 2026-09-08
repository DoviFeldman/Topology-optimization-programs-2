"""
Topology Optimization Studio — a simple web UI to run 3D SIMP topology
optimization, including on your own uploaded STL geometry.

Run it with:   streamlit run app.py   (or: bash run.sh)
In Codespaces the port is forwarded automatically; open the forwarded URL and
you get a clean UI in your browser.
"""

import tempfile

import numpy as np
import streamlit as st

from engines import simp3d, stl_io

st.set_page_config(page_title="Topology Optimization Studio", page_icon="🕸️", layout="wide")

st.title("🕸️ Topology Optimization Studio")
st.caption(
    "3D topology optimization you can run in the browser (Codespaces-friendly) "
    "or natively on your Mac. Optimize inside a box or inside your own uploaded "
    "STL, then download a printable STL. See `notes.txt` for the wider tool list."
)

page = st.sidebar.radio("Page", ["Programs (start here)", "3D SIMP → STL", "Other tools & how to run"])
st.sidebar.markdown("---")

FACES = ["x-", "x+", "y-", "y+", "z-", "z+"]
DIRS = ["-y", "+y", "-x", "+x", "-z", "+z"]


@st.cache_data(show_spinner=False)
def stl_to_domain_cached(data_bytes, resolution):
    import io
    return stl_io.stl_to_domain(io.BytesIO(data_bytes), resolution)


# --------------------------------------------------------------------------- #
# Programs overview  (how many, which work, where)
# --------------------------------------------------------------------------- #
if page.startswith("Programs"):
    st.header("How many programs, and where each one runs")
    st.markdown(
        "**There are 6 topology-optimization programs here.** "
        "**1** is built in and runs live in this app (including on your own STL). "
        "The other **5** are separate open-source tools — this repo tells you "
        "exactly how to install and run each, and where it works."
    )

    st.markdown(
        "| # | Program | Runs live in THIS app | GitHub Codespaces | Your Mac |\n"
        "|---|---------|:---:|:---:|:---:|\n"
        "| 1 | **3D SIMP** (built in) | ✅ yes | ✅ | ✅ |\n"
        "| 2 | **PyTopo3D** | ⛔ install & run separately | ✅ | ✅ |\n"
        "| 3 | **ToPy** | ⛔ install & run separately | ✅ | ✅ |\n"
        "| 4 | **OpenPISCO** | ⛔ install & run separately | ✅ | ✅ |\n"
        "| 5 | **Z88 Arion** (GUI) | ⛔ desktop app | ❌ | ❌ Windows only¹ |\n"
        "| 6 | **ToOptiX + FreeCAD** (GUI) | ⛔ desktop app | ❌ | ✅ |\n"
    )
    st.caption("¹ Z88 Arion is **Windows-only** in practice — the macOS build does "
               "not work (confirmed by testing).")

    st.markdown(
        "**In short:**\n"
        "- ✅ **Works right now, no setup:** the built-in **3D SIMP** — go to the "
        "**“3D SIMP → STL”** page and hit Run (upload your own STL there too).\n"
        "- ✅ **Run in Codespaces (Linux):** 3D SIMP, PyTopo3D, ToPy, OpenPISCO — "
        "**4 of 6**.\n"
        "- ✅ **Run on your Mac (Apple Silicon):** **5 of 6** — everything except "
        "Z88 Arion (Windows only). The GUI apps run on the desktop, not a browser.\n\n"
        "*(CalculiX is also documented as an FEA tool for **validating** a printed "
        "design — it's not a TO program itself.)*"
    )
    st.info("Upload-your-own-STL works in the built-in 3D SIMP (this app). PyTopo3D "
            "also imports STL via its own CLI; Z88/ToOptiX take geometry through "
            "FreeCAD. See the “Other tools & how to run” page for commands.")


# --------------------------------------------------------------------------- #
# 3D SIMP  (box OR uploaded STL) -> STL
# --------------------------------------------------------------------------- #
elif page.startswith("3D"):
    s = st.sidebar
    st.header("3D SIMP → STL")

    source = s.radio("Design domain", ["Box (built-in)", "Upload your own STL"])

    domain = None
    if source.startswith("Box"):
        nelx = s.slider("Elements X (nelx)", 12, 60, 32, 4)
        nely = s.slider("Elements Y (nely)", 8, 40, 16, 4)
        nelz = s.slider("Elements Z (nelz)", 8, 40, 16, 4)
        st.write("Optimizing inside a solid box. Pick supports, load, and a "
                 "material budget, then Run.")
    else:
        up = s.file_uploader("Upload an STL", type=["stl"])
        res = s.slider("Voxel resolution (longest side)", 16, 72, 40, 4,
                       help="Higher = finer detail but slower.")
        if up is None:
            st.info("⬆ Upload an STL in the sidebar. It becomes the **design "
                    "domain**: material is only ever placed inside your shape.")
            st.stop()
        with st.spinner("Voxelizing your STL…"):
            domain, meta = stl_to_domain_cached(up.getvalue(), res)
        nelx, nely, nelz = meta["nelx"], meta["nely"], meta["nelz"]
        st.success(f"Loaded your STL → {nelx}×{nely}×{nelz} voxel grid, "
                   f"{meta['filled']:,} solid voxels. Material will be optimized "
                   "inside this shape.")

    fixed_face = s.selectbox("Fixed (clamped) face", FACES, index=0)
    load_face = s.selectbox("Loaded face", FACES, index=1)
    load_dir = s.selectbox("Load direction", DIRS, index=0)
    volfrac = s.slider("Volume fraction", 0.10, 0.60, 0.30, 0.05)
    penal = s.slider("Penalization (SIMP)", 1.0, 5.0, 3.0, 0.5)
    rmin = s.slider("Filter radius (rmin)", 1.2, 4.0, 1.5, 0.1)
    iters = s.slider("Max iterations", 8, 60, 30, 2)
    threshold = s.slider("STL threshold", 0.2, 0.8, 0.5, 0.05)
    spacing = s.number_input("Voxel size (mm)", 0.5, 10.0, 1.0, 0.5)

    est = nelx * nely * nelz
    st.info(f"Grid = {nelx}×{nely}×{nelz} = {est:,} elements. 3D FE solves scale "
            "steeply and run on the server CPU — start modest; bigger = sharper "
            "but slower.")

    if st.button("▶ Run 3D optimization", type="primary"):
        prog = st.progress(0.0, text="starting…")

        def cb(it, ch, c, v):
            prog.progress(min(it / iters, 1.0),
                          text=f"iter {it} · compliance {c:.0f} · volume {v:.2f}")

        try:
            rho = simp3d.optimize(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac,
                                  penal=penal, rmin=rmin, max_iter=iters,
                                  domain=domain, fixed_face=fixed_face,
                                  load_face=load_face, load_dir=load_dir, callback=cb)
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            st.stop()
        prog.progress(1.0, text="building mesh…")

        tris = stl_io.voxel_mesh(rho, threshold=threshold, spacing=spacing)
        if len(tris) == 0:
            st.warning("Nothing above the threshold — lower the STL threshold or "
                       "raise the volume fraction.")
        else:
            try:
                import plotly.graph_objects as go
                V = tris.reshape(-1, 3)
                idx = np.arange(len(V)).reshape(-1, 3)
                fig = go.Figure(go.Mesh3d(
                    x=V[:, 0], y=V[:, 1], z=V[:, 2],
                    i=idx[:, 0], j=idx[:, 1], k=idx[:, 2],
                    color="#6ea8fe", opacity=1.0, flatshading=True))
                fig.update_layout(scene_aspectmode="data", height=600,
                                  margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.caption(f"(3D preview unavailable: {e})")

            tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
            stl_io.write_stl(tris, tmp.name)
            with open(tmp.name, "rb") as fh:
                st.download_button("⬇ Download STL", fh.read(),
                                   file_name="topopt3d.stl", mime="model/stl")
            st.caption(f"{len(tris):,} triangles · "
                       f"{int((rho >= threshold).sum()):,} solid voxels")


# --------------------------------------------------------------------------- #
# Other tools & how to run
# --------------------------------------------------------------------------- #
else:
    st.header("Other topology-optimization tools")
    st.write("The built-in **3D SIMP** always runs here. The tools below are the "
             "other 5 programs from `notes.txt`. Commands to install and run them:")

    with st.expander("PyTopo3D — pure-Python 3D SIMP with direct STL import/export", expanded=True):
        st.markdown("Codespaces ✅ · Mac ✅ — best match for a 3D-printing workflow.")
        st.code("pip install git+https://github.com/jihoonkim888/PyTopo3D.git\n"
                "# then follow the repo's examples (imports/exports STL directly)", language="bash")

    with st.expander("ToPy — Python topology optimization library"):
        st.markdown("Codespaces ✅ · Mac ✅")
        st.code("pip install topy   # 2D/3D SIMP driven by simple .tpd problem files", language="bash")

    with st.expander("OpenPISCO — feature-rich open-source platform"):
        st.markdown("Codespaces ✅ · Mac ✅ — overhang detection, STL export, thermal/buckling/modal.")
        st.code("conda install -c conda-forge openpisco   # or mamba", language="bash")

    with st.expander("Z88 Arion — free GUI topology optimization"):
        st.markdown("Codespaces ❌ (GUI) · **Windows only** (the macOS build does not "
                    "work in practice) — model/mesh in FreeCAD, optimize in Arion.")

    with st.expander("ToOptiX + FreeCAD addon"):
        st.markdown("Codespaces ❌ (GUI) · Mac ✅ — runs inside FreeCAD's GUI.")

    with st.expander("CalculiX (validation FEA — not a TO program)"):
        st.markdown("Codespaces ✅ · Mac ✅ — good for validating a printed design.")

    st.markdown("---")
    st.markdown("The full copied notes are in **`notes.txt`**.")
