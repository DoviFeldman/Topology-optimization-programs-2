"""
Topology Optimization Studio — a simple web UI to run the SIMP optimizers.

Run it with:   streamlit run app.py
In Codespaces the port is forwarded automatically; open the forwarded URL and
you get a clean UI in your browser — pick a program, set parameters, hit Run,
and view / download the result (PNG for 2D, STL for 3D printing).
"""

import io
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from engines import simp2d, simp3d, stl_io

st.set_page_config(page_title="Topology Optimization Studio", page_icon="🕸️", layout="wide")

st.title("🕸️ Topology Optimization Studio")
st.caption(
    "Self-contained SIMP topology optimization you can run right here in the "
    "browser (Codespaces-friendly) — no accounts, no external solver. "
    "See `notes.txt` for the wider landscape of TO tools."
)

program = st.sidebar.radio(
    "Program",
    ["2D SIMP (compliance)", "3D SIMP → STL", "Other tools & how to run"],
)
st.sidebar.markdown("---")


# --------------------------------------------------------------------------- #
# 2D SIMP
# --------------------------------------------------------------------------- #
if program.startswith("2D"):
    s = st.sidebar
    load = s.selectbox("Load case", ["mbb", "cantilever", "bridge"],
                       help="Boundary conditions + where the load is applied.")
    nelx = s.slider("Elements X (nelx)", 40, 240, 120, 10)
    nely = s.slider("Elements Y (nely)", 20, 120, 40, 5)
    volfrac = s.slider("Volume fraction", 0.10, 0.80, 0.50, 0.05,
                       help="Fraction of the domain kept as material.")
    penal = s.slider("Penalization (SIMP)", 1.0, 5.0, 3.0, 0.5)
    rmin = s.slider("Filter radius (rmin)", 1.2, 5.0, 2.4, 0.2,
                    help="Controls minimum feature size / mesh independence.")
    iters = s.slider("Max iterations", 10, 120, 60, 5)

    st.subheader(f"2D SIMP — {load} beam")
    st.write(
        "Minimizes structural compliance (maximizes stiffness) for the chosen "
        "material budget. Black = solid, white = void."
    )

    if s.button("▶ Run 2D optimization", type="primary"):
        prog = st.progress(0.0, text="starting…")
        img = st.empty()

        def draw(rho, title=None):
            fig, ax = plt.subplots(figsize=(10, 10 * nely / nelx))
            ax.imshow(-rho, cmap="gray", vmin=-1, vmax=0, interpolation="nearest")
            ax.axis("off")
            if title:
                ax.set_title(title)
            return fig

        def cb(it, ch, c, rho):
            prog.progress(min(it / iters, 1.0),
                          text=f"iter {it} · change {ch:.3f} · compliance {c:.1f}")
            if it == 1 or it % 3 == 0:
                fig = draw(rho)
                img.pyplot(fig)
                plt.close(fig)

        rho = simp2d.optimize(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal,
                              rmin=rmin, load=load, max_iter=iters, callback=cb)
        prog.progress(1.0, text="done")
        fig = draw(rho, f"{load} · volume {rho.mean():.2f}")
        img.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        st.download_button("⬇ Download PNG", buf.getvalue(),
                           file_name=f"topopt2d_{load}.png", mime="image/png")


# --------------------------------------------------------------------------- #
# 3D SIMP -> STL
# --------------------------------------------------------------------------- #
elif program.startswith("3D"):
    s = st.sidebar
    nelx = s.slider("Elements X (nelx)", 12, 60, 32, 4)
    nely = s.slider("Elements Y (nely)", 8, 40, 16, 4)
    nelz = s.slider("Elements Z (nelz)", 8, 40, 16, 4)
    volfrac = s.slider("Volume fraction", 0.10, 0.60, 0.30, 0.05)
    penal = s.slider("Penalization (SIMP)", 1.0, 5.0, 3.0, 0.5)
    rmin = s.slider("Filter radius (rmin)", 1.2, 4.0, 1.5, 0.1)
    iters = s.slider("Max iterations", 8, 60, 30, 2)
    threshold = s.slider("STL threshold", 0.2, 0.8, 0.5, 0.05,
                         help="Voxels with density above this become solid in the STL.")
    spacing = s.number_input("Voxel size (mm)", 0.5, 10.0, 1.0, 0.5)

    st.subheader("3D SIMP → STL")
    st.write(
        "A 3D cantilever (clamped on the x=0 face, loaded down at the far top "
        "edge). The result is thresholded into a printable STL you can download."
    )
    est = nelx * nely * nelz
    st.info(f"Grid = {nelx}×{nely}×{nelz} = {est:,} elements. "
            "3D FE solves scale steeply — start small (this runs on the server's "
            "CPU). Bigger grids = sharper parts but longer runs.")

    if s.button("▶ Run 3D optimization", type="primary"):
        prog = st.progress(0.0, text="starting…")

        def cb(it, ch, c, v):
            prog.progress(min(it / iters, 1.0),
                          text=f"iter {it} · compliance {c:.0f} · volume {v:.2f}")

        rho = simp3d.optimize(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac,
                              penal=penal, rmin=rmin, max_iter=iters, callback=cb)
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
                       f"{int((rho >= threshold).sum()):,} solid voxels · "
                       f"final volume {rho.mean():.2f}")


# --------------------------------------------------------------------------- #
# Other tools & how to run
# --------------------------------------------------------------------------- #
else:
    st.subheader("Other topology-optimization tools")
    st.write(
        "The two engines above are built in and always run here. The tools below "
        "are the ones from `notes.txt`. Those that run headless in Codespaces are "
        "installable in this same environment; the GUI/desktop ones are best on "
        "your Mac. See `README.md` for the full rundown."
    )

    with st.expander("PyTopo3D — pure-Python 3D SIMP with direct STL import/export", expanded=True):
        st.markdown(
            "Best match for a 3D-printing workflow. Runs in Codespaces or on macOS."
        )
        st.code("pip install git+https://github.com/jihoonkim888/PyTopo3D.git\n"
                "# then follow the repo's examples to run from a script/CLI", language="bash")

    with st.expander("ToPy — Python topology optimization library"):
        st.code("pip install topy\n# 2D/3D SIMP driven by simple .tpd problem files", language="bash")

    with st.expander("OpenPISCO — feature-rich open-source platform (conda)"):
        st.code("conda install -c conda-forge openpisco   # or mamba", language="bash")
        st.markdown("Overhang detection, STL export, thermal/buckling/modal analysis.")

    with st.expander("Desktop / GUI only (best on your Mac, not in a browser)"):
        st.markdown(
            "- **Z88 Arion** — free GUI TO app (has macOS/Unix builds).\n"
            "- **ToOptiX + FreeCAD addon** — runs inside FreeCAD's GUI.\n"
            "- **CalculiX + scripts** — powerful FEA when scripted; good for "
            "validating a printed design.\n\n"
            "These need a desktop GUI or a complex headless setup, so they aren't "
            "wired into this web app."
        )

    st.markdown("---")
    st.markdown("The full copied notes are in **`notes.txt`** in this repo.")
