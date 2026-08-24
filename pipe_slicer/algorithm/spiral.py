from pipe_slicer.types import Slice, Spiral
import numpy as np
import trimesh.path as path
import trimesh.path.entities as entities


def _normalizeRing(ring3D: np.ndarray, zeroPointIdx: int) -> np.ndarray:
    """
    Return an open ring (no duplicate closing point) rolled so the
    zero point (RMF reference intersection) is the first coordinate.
    """
    ring = ring3D
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]

    return np.roll(ring, -zeroPointIdx, axis=0)


def _reverseRing(ring: np.ndarray) -> np.ndarray:
    """Reverse traversal direction while keeping the zero point first."""
    return np.roll(ring[::-1], 1, axis=0)


def _ringAngles(ring: np.ndarray, frame) -> np.ndarray:
    """
    Azimuth of each ring vertex in the slice's RMF frame, unwrapped along
    the traversal and shifted so the zero point (index 0, which lies on the
    reference ray) is at angle 0. A full CCW loop ends near +2*pi, a CW
    loop near -2*pi.
    """
    rel = ring - frame.origin
    x = rel @ frame.reference
    y = rel @ frame.calcS()

    angles = np.unwrap(np.arctan2(y, x))
    return angles - angles[0]


def _parametrizeRing(ring: np.ndarray, frame) -> tuple[np.ndarray, np.ndarray]:
    """
    Orient the ring CCW in its RMF frame and return (ring, angles) where
    angles are monotonically non-decreasing azimuths from 0 at the zero
    point. Because consecutive RMF frames are minimal-twist, the same
    azimuth on consecutive rings is the twist-minimal correspondence at
    every point of the loop, not just at the zero point.
    """
    angles = _ringAngles(ring, frame)

    if angles[-1] < 0: # net clockwise traversal; flip to CCW
        ring = _reverseRing(ring)
        angles = _ringAngles(ring, frame)

    # sections are expected star-shaped about the centerline; clamp any
    # small local angle reversals so the parametrization stays monotone
    angles = np.maximum.accumulate(angles)

    return ring, angles


def _sampleRing(ring: np.ndarray, angles: np.ndarray, tVals: np.ndarray) -> np.ndarray:
    """
    Sample the closed ring at normalized azimuths tVals in [0, 1]
    (t * 2*pi radians around the RMF frame, measured from the zero point).
    """
    closed = np.vstack([ring, ring[0]])
    anglesClosed = np.append(angles, 2.0 * np.pi) # ring closes back at the zero point

    theta = np.clip(tVals, 0.0, 1.0) * 2.0 * np.pi

    sampled = np.column_stack([
        np.interp(theta, anglesClosed, closed[:, axis]) for axis in range(3)
    ])
    return sampled


def calcSpiralPath(
        slices: list[Slice],
        pointsPerLoop: int = 200
) -> Spiral:
    """
    Turn monotonically ordered slices into one continuous spiral path,
    the way vase mode works in slicers like OrcaSlicer.

    Points on consecutive rings are matched by azimuth angle in each
    slice's RMF frame. Since the frames are rotation-minimizing, equal
    azimuths on consecutive rings are the minimal-twist correspondence
    even when the cross-section shape changes along the tube (arc-length
    parametrization slips azimuthally there and shears the spiral).

    For each pair of consecutive slices the loop is traced from the
    zero point (t = 0) all the way around (t -> 1) while linearly
    blending from the current ring to the next:

        point(t) = (1 - t) * ring_i(t) + t * ring_{i+1}(t)

    At t = 1 the path lands exactly on ring_{i+1}'s zero point, which is
    where the next loop starts, so the whole path is seamless.

    Returns an (N, 3) array of ordered path points.
    """
    if len(slices) < 2:
        raise ValueError("Need at least two slices to build a spiral path")

    parametrized = [
        _parametrizeRing(_normalizeRing(s.ring3D, s.zeroPointIdx), s.frame)
        for s in slices
    ]

    tVals = np.linspace(0.0, 1.0, pointsPerLoop, endpoint=False)

    loops = []
    for idx in range(len(parametrized) - 1):
        ring, angles = parametrized[idx]
        ringNext, anglesNext = parametrized[idx + 1]

        current = _sampleRing(ring, angles, tVals)
        target = _sampleRing(ringNext, anglesNext, tVals)

        blend = tVals[:, np.newaxis]
        loops.append((1.0 - blend) * current + blend * target)

    # terminate the spiral exactly on the last ring's zero point
    loops.append(parametrized[-1][0][np.newaxis, 0])

    spiralPoints = np.vstack(loops)
    return Spiral(spiralPoints, pointsPerLoop)
