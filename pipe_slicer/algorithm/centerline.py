import trimesh
import trimesh.path as path
import scipy.interpolate
import numpy as np
import shapely
from pipe_slicer.types import SegmentedMesh, Centerline




def _sectionToPath3D(
        surface:trimesh.Trimesh, 
        plane_origin: np.ndarray, 
        plane_normal: np.ndarray) -> path.path.Path3D:
    
        wall_section = surface.section(plane_origin=plane_origin, plane_normal=plane_normal)

        if wall_section is not None:
            assert isinstance(wall_section, path.path.Path3D)
        else:
            raise TypeError("Zero Intersections")

        return wall_section
    

def _applyTransform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]

def _ensurePolygonOutput(result: list) -> list[shapely.geometry.Polygon]:
    for p in result:
            assert isinstance(p, shapely.geometry.Polygon)
    return result



def _sectionCentroid(
        surface: trimesh.Trimesh,
        origin: np.ndarray,
        normal: np.ndarray
) -> np.ndarray | None:
    """
    Centroid of the cross-section polygon nearest to `origin`, or None if
    the plane misses the surface / yields no closed polygon.
    """
    try:
        intersection = _sectionToPath3D(surface, origin, normal)
    except TypeError:
        return None

    planar_intersection, transform = intersection.to_2D()

    polygons = planar_intersection.polygons_full
    if polygons is None or len(polygons) == 0:
        return None

    pCentroids3D = _applyTransform(
        transform,
        np.array([[p.centroid.x, p.centroid.y, 0] for p in polygons])
    )

    nearness = np.argsort(np.linalg.norm(pCentroids3D - origin, axis=1))
    return pCentroids3D[nearness[0]]


def _marchSections(
        surface: trimesh.Trimesh,
        starting_point: np.ndarray,
        starting_normal: np.ndarray,
        step_dist: float,
        max_steps: int = 10000,
        corrector_iters: int = 2
    ) -> np.ndarray:


    ### starting values for marching algorithm ###
    position: np.ndarray = starting_point
    unit_tangent: np.ndarray = starting_normal
    centroids = [position] # seed with original centroid

    ### normal must point towards the tube ###
    appx_tube_dir = surface.centroid - starting_point
    dot = np.dot(appx_tube_dir, unit_tangent)
    mags = np.linalg.norm(appx_tube_dir) * np.linalg.norm(unit_tangent)

    cos_angle = dot / mags

    if cos_angle <= 0: # unit normal is pointing the wrong direction
        unit_tangent = unit_tangent * -1

    for step in range(max_steps):
        # predictor: step along the current tangent, then correct the
        # plane normal toward the secant to the found centroid so the cut
        # stays near-orthogonal on bends (a lagging normal cuts obliquely
        # and drags the centroid off the true axis)
        origin = position + step_dist * unit_tangent

        centroid = None
        for _ in range(1 + corrector_iters):
            found = _sectionCentroid(surface, origin, unit_tangent)
            if found is None:
                break
            centroid = found

            tangent = centroid - position
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-9:
                return np.array(centroids)

            unit_tangent = tangent / tangent_norm
            origin = position + step_dist * unit_tangent

        if centroid is None:
            break

        position = centroid
        centroids.append(centroid)

    return np.array(centroids)


def _refineCenterline(
        surface: trimesh.Trimesh,
        bspl: scipy.interpolate.BSpline,
        sample_dist: float,
        degree: int,
        passes: int,
        smooth: float = 0.0,
        anchors: tuple[np.ndarray, np.ndarray] | None = None
) -> tuple[scipy.interpolate.BSpline, np.ndarray]:
    """
    Iteratively re-derive the centerline from itself: sample the current
    spline uniformly in arc length, cut the surface orthogonally using the
    spline tangent, and refit through the section centroids. Orthogonal
    cuts of a (locally) axisymmetric tube put the centroid on the true
    axis, so each pass removes the lag/divergence the coarse march bakes in.
    """
    rawPoints = np.array([])

    for _ in range(passes):
        deriv = bspl.derivative(1)

        uDense = np.linspace(0.0, 1.0, 5000)
        coordsDense = np.column_stack(bspl(uDense))
        lengths = np.concatenate([
            [0.0], np.cumsum(np.linalg.norm(np.diff(coordsDense, axis=0), axis=1))
        ])

        # stay off the extreme ends: the spline can overshoot there (endpoint
        # oscillation, especially at higher degree) and grazing the tube rim
        # yields degenerate sections
        margin = 0.5 * sample_dist
        sVals = np.arange(margin, lengths[-1] - margin, sample_dist)
        uVals = np.interp(sVals, lengths, uDense)

        centroids = []
        for u in uVals:
            origin = np.asarray(bspl(u))
            tangent = np.asarray(deriv(u))
            tangent = tangent / np.linalg.norm(tangent)

            centroid = _sectionCentroid(surface, origin, tangent)
            if centroid is None:
                continue

            # reject teleporting outliers (section picked a far-away polygon);
            # compare against the sample origin, never against the previous
            # accepted point — a relative check cascades after one rejection
            if np.linalg.norm(centroid - origin) > 2.0 * sample_dist:
                continue

            centroids.append(centroid)

        if anchors is not None:
            # end-face centroids sit exactly on the tube axis; anchoring the
            # fit there keeps the spline covering the full tube (the sampled
            # centroids stop a margin short of the rims)
            centroids = [anchors[0]] + centroids + [anchors[1]]

        if len(centroids) <= degree:
            break

        rawPoints = np.array(centroids)
        bspl = _genSpline(rawPoints, degree, smooth)

        # robust refit: sections that cut across a radius step (bulge/groove
        # edges) get laterally biased centroids the smooth fit cannot
        # reconcile; cull them and refit through the trustworthy points
        uu = np.linspace(0.0, 1.0, 4000)
        cc = np.column_stack(bspl(uu))
        residuals = np.array([
            np.min(np.linalg.norm(cc - p, axis=1)) for p in rawPoints
        ])
        threshold = max(3.0 * float(np.median(residuals)), 0.15)
        keep = residuals < threshold
        keep[0] = keep[-1] = True # never drop the end anchors

        if (~keep).any() and keep.sum() > degree:
            rawPoints = rawPoints[keep]
            bspl = _genSpline(rawPoints, degree, smooth)

    return bspl, rawPoints


        


def _genSpline(rawCL: np.ndarray, degree: int, smooth: float = 0.0) -> scipy.interpolate.BSpline:
    x = [rawCL[:, 0], rawCL[:, 1], rawCL[:, 2]]
    bspl, params = scipy.interpolate.make_splprep(x, k=degree, s=smooth)
    return bspl


def calcCenterline(
        sm: SegmentedMesh,
        wall: str="inner",
        k=5,
        march_step: float = 5.0,
        refine_dist: float = 2.0,
        refine_passes: int = 2,
        smooth: float = 2.0
) -> Centerline:
    """
    Compute the tube centerline: coarse section-marching, then refinement
    passes that re-slice orthogonally along the fitted spline and refit
    through the section centroids.

    `smooth` is the spline smoothing factor (scipy `s`); a small positive
    value rejects tessellation noise in the centroids and, critically,
    suppresses endpoint oscillation of the interpolating fit, which
    otherwise leaves the end tangents pointing off-axis. Degrees above 5
    are prone to oscillation even with smoothing.
    """

    if wall == "inner":
        surf = sm.inner
    elif wall == "outer":
        surf = sm.outer
    else:
        raise TypeError(f"Undefined surface {wall}")

    starting_point = sm.bottom.centroid
    starting_normal = sm.bottom.facets_normal[0]

    rawLine = _marchSections(surf, starting_point, starting_normal, march_step)
    bspl = _genSpline(rawLine, k, smooth)

    if refine_passes > 0:
        anchors = (np.asarray(sm.bottom.centroid), np.asarray(sm.top.centroid))
        bspl, refined = _refineCenterline(
            surf, bspl, refine_dist, k, refine_passes, smooth, anchors)
        if refined.size > 0:
            rawLine = refined

    return Centerline(bspl, rawLine)

    


    
    
