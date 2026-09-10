"""Quick matplotlib renders of a mesh (headless, no GL needed)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render_mesh(verts, faces, path, title="", views=((18, -70), (18, 20), (78, -90)),
                color="#6fa8dc"):
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces)
    fig = plt.figure(figsize=(4.2 * len(views), 5.2), facecolor="white")
    light = np.array([0.4, 0.3, 0.86]); light /= np.linalg.norm(light)
    base = np.array(matplotlib.colors.to_rgb(color))
    tris = verts[faces]
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True); ln[ln == 0] = 1
    nrm = nrm / ln
    shade = 0.30 + 0.70 * np.clip(np.abs(nrm @ light), 0, 1)
    cols = np.clip(shade[:, None] * base[None, :], 0, 1)
    lo, hi = verts.min(0), verts.max(0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2 * 1.03
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors="none"))
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    fig.suptitle("%s   (%d tris)" % (title, len(faces)), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return path
