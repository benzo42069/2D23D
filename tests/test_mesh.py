import numpy as np

from core.mesh import depth_to_mesh


def test_depth_to_mesh_generates_faces():
    depth = np.array(
        [
            [0.1, 0.2, 0.2],
            [0.15, 0.3, 0.25],
            [0.2, 0.25, 0.2],
        ],
        dtype=np.float32,
    )
    mask = np.ones((3, 3), dtype=np.uint8)
    mesh = depth_to_mesh(depth, mask, scale=1.0, thickness=0.05, smooth_iterations=0)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) >= 8
    assert mesh.vertices[:, 2].max() > mesh.vertices[:, 2].min()
