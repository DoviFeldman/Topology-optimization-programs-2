"""
Topology Optimization Studio v2.

    bash run.sh          # then open http://localhost:8501

Upload an STL, pick a look, say how the part is held and loaded, press a
button. A quick preview takes well under a minute; the final a few minutes.
"""
from __future__ import annotations

import io
import os
import time

import numpy as np
import streamlit as st

from engines import pipeline, voxel

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
UPLOADS = os.path.join(ROOT, "uploads")
RESULTS = os.path.join(ROOT, "results")
for d in (UPLOADS, RESULTS):
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title="Topology Optimization Studio", layout="wide")


def figure(mesh, color):
    import plotly.graph_objects as go
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    fig = go.Figure(go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.15, roughness=0.7),
        lightposition=dict(x=300, y=-400, z=600)))
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(aspectmode="data", xaxis_visible=False,
                                 yaxis_visible=False, zaxis_visible=False,
                                 camera=dict(eye=dict(x=1.5, y=-1.7, z=1.0))))
    return fig


st.title("Topology Optimization Studio")
left, right = st.columns([1, 1.6], gap="large")

with left:
    st.subheader("1 · Part")
    up = st.file_uploader("Upload an STL", type=["stl"])
    samples = sorted(f for f in os.listdir(SAMPLES)) if os.path.isdir(SAMPLES) else []
    stl_path = None
    if up is not None:
        stl_path = os.path.join(UPLOADS, up.name)
        with open(stl_path, "wb") as fh:
            fh.write(up.getbuffer())
    elif samples:
        pick = st.selectbox("...or try a sample", samples)
        stl_path = os.path.join(SAMPLES, pick)
    else:
        st.caption("No samples yet - run `python make_test_shapes.py` to make some.")
    up_axis = st.selectbox("Which axis of the STL points up?", ["z", "y", "x"])

    st.subheader("2 · Look")
    style = st.radio("Style", list(pipeline.STYLES), index=1, horizontal=True)
    st.caption(pipeline.STYLES[style]["blurb"])

    st.subheader("3 · How it is held and loaded")
    load = st.selectbox("Setup", list(pipeline.LOADS))
    st.caption(pipeline.LOADS[load]["blurb"])

    with st.expander("Fine-tune"):
        sty = pipeline.STYLES[style]
        volfrac = st.slider("Material kept (fraction of the solid part)",
                            0.06, 0.60, float(sty["volfrac"]), 0.01)
        member = st.slider("Member size (fraction of the longest side)",
                           0.03, 0.25, float(sty["member"]), 0.01,
                           help="Bigger = fewer, thicker limbs. Small sizes with "
                                "little material give a perforated skin instead "
                                "of struts.")
        pad = st.slider("Solid pad under loads and supports", 0.0, 0.10, 0.035, 0.005)
        detail = st.radio("Detail of the final run", ["standard", "fine"],
                          horizontal=True,
                          help="standard: ~1-3 min. fine: ~5-10 min, only worth "
                               "it for skeleton and lace.")

    c1, c2 = st.columns(2)
    go_prev = c1.button("Quick preview", disabled=stl_path is None,
                        use_container_width=True)
    go_final = c2.button("Final", type="primary", disabled=stl_path is None,
                         use_container_width=True)

with right:
    if (go_prev or go_final) and stl_path:
        level = detail if go_final else "preview"
        bar = st.progress(0.0, "voxelizing...")
        budget = {"preview": 30, "standard": 45, "fine": 45}[level]

        def prog(h):
            bar.progress(min(h["it"] / budget, 0.97),
                         "%s pass - step %d - %.0f s" % (h["level"], h["it"], h["seconds"]))

        try:
            mesh, info, _, _ = pipeline.run(
                stl_path, up_axis=up_axis, style=style, load=load, detail=level,
                volfrac=volfrac, member=member, pad_thickness=pad, progress=prog)
        except Exception as exc:
            bar.empty()
            st.error(str(exc))
            st.stop()
        bar.empty()
        name = "%s_%s_%s_%s" % (os.path.splitext(os.path.basename(stl_path))[0],
                                style, load, level)
        buf = io.BytesIO()
        mesh.export(buf, file_type="stl")
        with open(os.path.join(RESULTS, name + ".stl"), "wb") as fh:
            fh.write(buf.getvalue())
        st.session_state["result"] = dict(mesh=mesh, info=info, name=name,
                                          stl=buf.getvalue())

    res = st.session_state.get("result")
    if res:
        info, m = res["info"], res["info"]["mesh"]
        st.success("%s - done in %.0f s" % (res["name"], info["seconds"]))
        a, b, c, d = st.columns(4)
        a.metric("Material kept", "%.0f %%" % (100 * (info["mass_kept"] or 0)))
        b.metric("Triangles", format(m["faces"], ","))
        c.metric("Watertight", "yes" if m["watertight"] else "NO")
        d.metric("Voxel", "%.2f mm" % info["voxel_mm"])
        if m["pieces"] > m["pieces_kept"]:
            st.warning("%d loose fragments were dropped. Try a larger member "
                       "size or more material." % (m["pieces"] - m["pieces_kept"]))
        if not m["exact_outer"]:
            st.info("The input STL is not a clean closed solid, so the outer "
                    "faces were rebuilt from voxels instead of cut exactly.")
        st.plotly_chart(figure(res["mesh"], "#c9ccd1"), use_container_width=True)
        st.download_button("Download STL", res["stl"], file_name=res["name"] + ".stl",
                           mime="model/stl", type="primary")
    elif stl_path:
        st.plotly_chart(figure(voxel.load_mesh(stl_path), "#9aa5b1"),
                        use_container_width=True)
        st.caption("Your part. Pick a look and press Quick preview.")
