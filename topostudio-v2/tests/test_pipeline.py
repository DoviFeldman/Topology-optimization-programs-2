"""Smoke tests:  python -m pytest tests -q   (about a minute)."""
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines import pipeline, topopt, voxel  # noqa: E402


def _box(ext):
    b = trimesh.creation.box(extents=ext)
    b.apply_translation(np.array(ext) / 2.0)
    return b


def test_voxelizer_is_exact_for_a_box():
    occ, g = voxel.voxelize(_box((40, 20, 30)), 40)
    assert occ.shape == (40, 20, 30) and occ.all()


def test_element_passes_rigid_body_check():
    KE = topopt.hex8_KE()
    t = np.tile([1.0, 0, 0], 8)
    assert abs(KE @ t).max() < 1e-12 and abs(KE - KE.T).max() < 1e-12


def test_preview_run_is_printable_and_keeps_outer_size():
    part = _box((40, 40, 60))
    mesh, info, _, _ = pipeline.run(part, style="organic", load="stand",
                                    detail="preview")
    assert mesh.is_watertight
    assert info["mesh"]["exact_outer"]
    assert np.allclose(mesh.extents, part.extents, atol=1e-6)
    assert 0.08 < info["mass_kept"] < 0.30
    # converged, not flip-flopping: the design is barely moving at the end
    assert info["history"][-1]["change"] < 0.02
