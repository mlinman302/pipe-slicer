from pipe_slicer.types import Spiral, FlowSpiral, FlowCharacteristics
import numpy as np


class WindingSpacingError(ValueError):
    """
    Raised when the actual distance between adjacent windings makes the
    part unprintable: windings colliding on the inside of a bend, or
    pulled too far apart to fuse on the outside.
    """


def _pointToSegmentDist(
        points: np.ndarray,
        segStarts: np.ndarray,
        segEnds: np.ndarray
) -> np.ndarray:
    """
    Distance from each points[i] to the segment segStarts[i] -> segEnds[i].
    All inputs (M, 3), returns (M,).
    """
    seg = segEnds - segStarts
    segLenSq = np.einsum("ij,ij->i", seg, seg)

    # parameter of the projection onto each segment, clamped to the segment
    t = np.einsum("ij,ij->i", points - segStarts, seg)
    t = t / np.where(segLenSq < 1e-12, 1.0, segLenSq)
    t = np.clip(t, 0.0, 1.0)

    proj = segStarts + t[:, np.newaxis] * seg
    return np.linalg.norm(points - proj, axis=1)


def _describeLocation(pointIdx: int, pointsPerLoop: int, h: float) -> str:
    """Human-readable location of a path point: arc length s and clock position."""
    loopIdx = pointIdx // pointsPerLoop
    tFrac = (pointIdx % pointsPerLoop) / pointsPerLoop

    s = (loopIdx + tFrac) * h  # approx arc length along centerline from first winding
    clockHours = (tFrac * 12.0) % 12.0
    if clockHours < 0.5:
        clockHours = 12.0

    return f"s = {s:.2f} (winding {loopIdx}), clock position {clockHours:.1f} o'clock"


def calcFlow(
        spiral: Spiral,
        h: float,
        searchWindow: int = 3,
        minFlow: float = 0.3,
        maxFlow: float = 2.0,
        collisionRatio: float = 0.25,
        gapRatio: float = 1.75
) -> FlowSpiral:
    """
    Compute per-point winding spacing and extrusion flow compensation.

    On a bend the windings on the inside of the curve sit closer together
    than the nominal spacing `h`, and on the outside farther apart; constant
    extrusion would over-stuff the inside and under-fill the outside.

    For each path point Q_i the previous winding is the path one full turn
    (theta - 2*pi) below, i.e. pointsPerLoop samples earlier. The spacing is
    the distance from Q_i to that winding, refined by projecting onto the
    previous winding's local segments (+/- searchWindow segments around the
    matching sample). To first order this equals h * (1 - kappa(s) * rho)
    where rho is the point's signed offset from the centerline toward the
    inside of the bend.

    Flow multiplier is spacing / h, clamped to [minFlow, maxFlow]. Points on
    the first winding have nothing below them and get spacing = h, flow = 1.

    Raises WindingSpacingError when the actual path violates printability:
    - spacing < collisionRatio * h: windings colliding, inner bend too tight
    - spacing > gapRatio * h: adjacent windings no longer fuse, gap in the wall.
    """
    points = spiral.points
    numPoints = points.shape[0]
    pointsPerLoop = spiral.pointsPerLoop

    if numPoints <= pointsPerLoop:
        raise ValueError("Spiral has less than one full winding; nothing to space against")

    spacing = np.full(numPoints, h, dtype=float)

    # points that have a winding below them
    idx = np.arange(pointsPerLoop, numPoints)
    query = points[idx]

    # candidate segments on the previous winding, centered on the sample at theta - 2*pi
    best = np.full(idx.shape[0], np.inf)
    for offset in range(-searchWindow, searchWindow):
        segStartIdx = np.clip(idx - pointsPerLoop + offset, 0, numPoints - 2)
        dist = _pointToSegmentDist(query, points[segStartIdx], points[segStartIdx + 1])
        best = np.minimum(best, dist)

    spacing[idx] = best

    ### hard printability checks on the actual path ###
    collisions = idx[best < collisionRatio * h]
    if collisions.size > 0:
        worst = collisions[np.argmin(spacing[collisions])]
        raise WindingSpacingError(
            f"Windings colliding: inner bend too tight at {collisions.size} point(s). "
            f"Worst spacing {spacing[worst]:.3f} (< {collisionRatio:.2f} * h = {collisionRatio * h:.3f}) "
            f"at {_describeLocation(int(worst), pointsPerLoop, h)}"
        )

    gaps = idx[best > gapRatio * h]
    if gaps.size > 0:
        worst = gaps[np.argmax(spacing[gaps])]
        raise WindingSpacingError(
            f"Adjacent windings no longer fuse: wall gap at {gaps.size} point(s). "
            f"Worst spacing {spacing[worst]:.3f} (> {gapRatio:.2f} * h = {gapRatio * h:.3f}) "
            f"at {_describeLocation(int(worst), pointsPerLoop, h)}"
        )

    rawFlow = spacing / h
    flow = np.clip(rawFlow, minFlow, maxFlow)

    chars = FlowCharacteristics(
        flowMin=float(np.min(flow)),
        flowMax=float(np.max(flow)),
        flowMean=float(np.mean(flow)),
        stdDev=float(np.std(flow)),
        clampedFraction=float(np.mean((rawFlow < minFlow) | (rawFlow > maxFlow)))
    )

    return FlowSpiral(spiral=spiral, h=h, spacing=spacing, flow=flow, chars=chars)
