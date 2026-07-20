from pipe_slicer.algorithm.centerline import _sectionToPath3D
from pipe_slicer.types import Centerline, Frame, Slice
import numpy as np
import trimesh
import trimesh.path as path
import shapely.geometry as geometry



def _insertPointOnRing(
        ringCoords: np.ndarray,
        point: np.ndarray,
        minDist: float = 1e-9
) -> tuple[np.ndarray, int]:
    """
    insert a new point into a ring on the existing edge. Point given must preexist on the edge
    (ie, determined from an intersection)
    """

    # remove last point (as it is a copy of the first to form a ring)
    numPoints = len(ringCoords) - 1

    for pointIdx in range(numPoints):
        coord = ringCoords[pointIdx]
        nextCoord = ringCoords[pointIdx + 1]

        lineSeg = nextCoord - coord
        lineSegSq = lineSeg @ lineSeg # dot

        # if line segment is very very very close to length 0
        if lineSegSq < minDist:
            continue
        
        # amount along line segment the projection of t is
        t = ((point - coord) @ lineSeg) / lineSegSq

        # check if within the segment
        if -minDist <= t <= 1 + minDist: # minDist added for tolerance
            proj = coord + np.clip(t, 0, 1) * lineSeg
            if np.linalg.norm(proj - point) < minDist:
                newRing = np.insert(ringCoords, pointIdx + 1, point, axis=0)
                return newRing, pointIdx + 1
            
    raise ValueError("Point does not lie on any segment of polygon")



def _findRadialIntersection(
        origin3D: np.ndarray,
        radialDir3D: np.ndarray,
        polygon: geometry.Polygon,
        transformTo3D: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    returns (hitPoint3D, updated2DRing, insertedPointIdx)
    """
    transformTo2D = np.linalg.inv(transformTo3D)

    # full transform
    originH = np.append(origin3D, 1.0)
    origin2D = (transformTo2D @ originH)[:2]

    # just rotation
    dir2D = (transformTo2D[:3, :3] @ radialDir3D)[:2]
    dir2DUnit = dir2D / np.linalg.norm(dir2D)

    minx = polygon.bounds[0]
    miny = polygon.bounds[1]
    maxx = polygon.bounds[2]
    maxy = polygon.bounds[3]

    maxDiag = max(maxx - minx, maxy - miny)

    rayEnd = origin2D + dir2D * maxDiag * 2

    ray = geometry.LineString([origin2D, rayEnd])

    hit = ray.intersection(polygon.exterior)

    if hit.is_empty:
        raise ValueError("No intersection between frame normal and mesh section")
    
    
    if not isinstance(hit, geometry.Point):
        raise ValueError("Intersection yielded non-point geoemetry")

    hit2D = np.array(hit.coords[0])

    updatedRing, intersectionIdx = _insertPointOnRing(np.array(polygon.exterior.coords), hit2D)

    hitH = np.array([hit2D[0], hit2D[1], 0.0, 1.0])
    hit3D = (transformTo3D @ hitH)[:3]

    return hit3D, updatedRing, intersectionIdx




def _ensurePolygonList(polygonsFull) -> list[geometry.Polygon]:
    """Normalize trimesh's polygons_full (ndarray of Polygon, tuple, or None) to a list[Polygon]."""
    if polygonsFull is None:
        return []
    
    result = list(polygonsFull)

    for p in result:
        if not isinstance(p, geometry.Polygon):
            raise TypeError(f"Expected shapely Polygon, got {type(p)}")
        
    return result

def _getClosestPolygonIntersection(
        path: path.path.Path2D,
        origin3D: np.ndarray,
        transformTo3D: np.ndarray
):
    """
    Pick the section polygon whose centroid is nearest the slice origin
    (the centerline point). A plane can cut a curved tube more than once;
    the extra rings are far from the centerline, not from the section's
    overall centroid.
    """
    polygons: list[geometry.Polygon] = _ensurePolygonList(path.polygons_closed)

    if not polygons:
        raise ValueError("No polygons found in cross-section")

    origin2D = (np.linalg.inv(transformTo3D) @ np.append(origin3D, 1.0))[:2]

    return min(
        polygons,
        key=lambda p: np.dot(p.centroid.coords[0] - origin2D, p.centroid.coords[0] - origin2D)
    )


def _initialReference(tangent: np.ndarray) -> np.ndarray:
    """
    Build an arbitrary reference vector orthogonal to the first tangent,
    used to seed the very first frame (no previous frame to propagate from).
    """
    # pick whichever world axis is least parallel to tangent, to avoid
    # a near-zero cross product
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangent, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    reference = world_up - np.dot(world_up, tangent) * tangent
    return reference / np.linalg.norm(reference)


def _calcRMF(
    prevFrame: Frame | None,
    coords: np.ndarray,
    tangent: np.ndarray,
) -> Frame:
    """
    Propagate a rotation-minimizing frame from prevFrame (at prevCoord) to
    the new point `coords` with tangent `tangent`, via the double
    reflection method (Wang et al. 2008).
    """
    if prevFrame is None:
        return Frame(tangent=tangent, reference=_initialReference(tangent), origin=coords)

    t_prev = prevFrame.tangent
    r_prev = prevFrame.reference

    # --- first reflection: through the plane bisecting prevCoord -> coords ---
    v1 = coords - prevFrame.origin
    c1 = v1 @ v1
    if c1 < 1e-12:
        # points coincide; nothing to propagate, carry frame forward as-is
        return Frame(tangent=tangent, reference=r_prev, origin=coords)

    r_L = r_prev - (2.0 / c1) * (v1 @ r_prev) * v1
    t_L = t_prev - (2.0 / c1) * (v1 @ t_prev) * v1

    # --- second reflection: through the plane bisecting t_L -> tangent ---
    v2 = tangent - t_L
    c2 = v2 @ v2
    if c2 < 1e-12:
        # tangent didn't change direction; reference survives first reflection unchanged
        r_new = r_L
    else:
        r_new = r_L - (2.0 / c2) * (v2 @ r_L) * v2

    r_new = r_new / np.linalg.norm(r_new)

    return Frame(tangent=tangent, reference=r_new, origin=coords)


def _getUfromS(s: float, uVals: np.ndarray, lengthsAtU: np.ndarray) -> float:
    return float(np.interp(s, lengthsAtU, uVals))


def calcSlices(
        cl: Centerline,
        mesh: trimesh.Trimesh,
        step_dist: float,
        max_steps: int = 10000
    ) -> list[Slice]:


    bspl = cl.spline
    deriv = bspl.derivative(nu=1)

    ### as we are interested in looking up the parameter by the arc length ###
    uDense = np.linspace(0, 1, 5000)
    coordsDense = np.column_stack(bspl.__call__(uDense))
    lineSegLengths = np.linalg.norm(np.diff(coordsDense, axis=0), axis=1)
    lengthsAtU = np.concatenate([[0], np.cumsum(lineSegLengths)]) # idx corresponds to U
    totalArcLength = lengthsAtU[-1]

    print(f"Total arc length: {totalArcLength}")


    steps = np.arange(step_dist, totalArcLength, step_dist)

    prevFrame = None

    slices: list[Slice] = []

    for step in steps:
        u = _getUfromS(step, uDense, lengthsAtU)

        coord = bspl.__call__(u)
        tangent = deriv.__call__(u)
        unitTangent = tangent / np.linalg.norm(tangent)

        frame = _calcRMF(prevFrame, coord, unitTangent)

        # a near-orthogonal plane this close to a rim can graze the mesh
        # boundary and yield an open (or no) section, and the smoothed
        # centerline can overshoot the rims slightly; skip those instead of
        # failing, but only at the extreme ends of the tube
        endMargin = max(2 * step_dist, 2.0)
        nearEnd = step < endMargin or step > totalArcLength - endMargin

        ### calculate intersection of plane and mesh ###
        wall_section = mesh.section(plane_origin=coord, plane_normal=unitTangent)
        if wall_section is not None:
            assert isinstance(wall_section, path.path.Path3D)
        elif nearEnd:
            continue
        else:
            raise TypeError("Zero Intersections")
        planar_intersection, transform = wall_section.to_2D()


        # get the proper polygon section
        try:
            section = _getClosestPolygonIntersection(planar_intersection, coord, transform)
        except ValueError:
            if nearEnd:
                continue
            raise

        # insert the RMF reference into the polygon
        try:
            polygonStartPoint3D, polygon2D, insertedPointIdx = _findRadialIntersection(coord, frame.reference, section, transform)
        except ValueError as v:
            if nearEnd:
                continue
            raise ValueError(f"{v} at step dist {step}")

        numCoords = polygon2D.shape[0]
        polygon2DH = np.column_stack([polygon2D, np.zeros(numCoords), np.ones(numCoords)])
        transformed = polygon2DH @ transform.T
        polygon3D = transformed[:, :3]

        currentSlice = Slice(polygon3D, insertedPointIdx, frame)

        slices.append(currentSlice)

        prevFrame = frame

    return slices






