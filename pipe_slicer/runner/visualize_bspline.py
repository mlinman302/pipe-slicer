"""
Short runner to visualize a scipy BSpline curve using trimesh.

Usage:
    Fit or load your BSpline (`spl`), then call `visualize_bspline(spl)`.
"""

import numpy as np
import trimesh
from scipy.interpolate import BSpline


def visualize_bspline(spl: BSpline, count: int = 200, mesh: trimesh.Trimesh | None = None) -> None:
    """
    Sample a scipy BSpline and render it as a line path in a trimesh scene.

    spl:   scipy.interpolate.BSpline object (vector-valued, e.g. 3D)
    count: number of samples along the curve
    mesh:  optional trimesh.Trimesh to render alongside the curve
    """
    # sample the spline across its valid parameter domain
    u = np.linspace(spl.t[spl.k], spl.t[-spl.k - 1], count)
    points = spl(u)

    # normalize to shape (count, ndim) regardless of how the spline stores axis order
    # (some spline constructions return (ndim, count) instead)
    if points.ndim == 2 and points.shape[0] in (2, 3) and points.shape[1] == count:
        points = points.T

    # build line segments between consecutive sampled points
    line_segs = np.stack([points[:-1], points[1:]], axis=1)  # shape (count-1, 2, 3)

    path = trimesh.load_path(line_segs)
    path.colors = np.tile([0, 255, 0, 255], (len(path.entities), 1))  # green

    scene = trimesh.Scene()
    if mesh is not None:
        scene.add_geometry(mesh, node_name="mesh")
    scene.add_geometry(path, node_name="bspline")

    scene.show()


if __name__ == "__main__":
    # Example usage — replace with your actual fitted spline / mesh
    from scipy.interpolate import make_splprep

    # dummy 3D points to fit a spline for demonstration
    t = np.linspace(0, 2 * np.pi, 20)
    demo_points = [np.cos(t), np.sin(t), t]  # x, y, z

    spl, u = make_splprep(demo_points)

    visualize_bspline(spl)
