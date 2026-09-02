from pipe_slicer.types import FlowSpiral, GCodeConfig, GCodeProgram, PurgeConfig
import numpy as np
import matplotlib.pyplot as plt

def projectOntoXZ(vectors: np.ndarray) -> np.ndarray:
    """
    Orthogonally project row vectors onto the XZ plane by dropping Y.

    The B axis rotates the head about +Y, so the only orientations it can
    reach lie in the XZ plane. Projecting is what the machine physically
    does to a requested direction: the Y component is simply unreachable,
    and dropping it leaves the part of the vector B can actually track,
    with the signs of both x and z intact.

    Note this is a projection, not a rotation - a vector leaning out of the
    XZ plane comes back shorter. The length it loses, asin(|y| / |v|), is
    exactly the orientation error the B axis cannot remove.
    """
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    return np.column_stack((x, np.zeros_like(y), z))


def calcBAngle(tangents: np.ndarray) -> np.ndarray:
    """
    B axis angle, in degrees, that points the nozzle along each centerline
    tangent.

    B rotates the head about +Y, aiming the nozzle along

        n(B) = (sin B, 0, cos B)

    so the best B for a tangent t is the one maximising n(B) . t, i.e.
    sin B * tx + cos B * tz, giving

        B = degrees(atan2(tx, tz))

    which is the signed angle of the tangent projected onto XZ (see
    projectOntoXZ), measured from +Z towards +X. Two properties matter:

    - It is signed. A tangent leaning +X gives positive B and one leaning
      -X gives negative B, so the head tilts the way the tube actually
      leans. Taking a magnitude (arccos, or hypot(tx, ty)) collapses both
      onto one sign and tilts the head backwards over half of an S-bend.
    - atan2 keeps the sign of tz, so the angle runs continuously through
      the horizontal (B = +/-90) and on to +/-180 for a tangent aimed
      downwards, instead of folding back toward zero there.

    A tangent leaning out of the XZ plane cannot be tracked by B alone;
    this returns the closest reachable angle and the residual misalignment
    is asin(|ty|). Removing it needs a bed/head rotation about Z, which the
    emitter does not model yet.
    """
    return np.degrees(np.arctan2(tangents[:, 0], tangents[:, 2]))


def calcFeedrates(
        numPoints: int,
        pointsPerLoop: int,
        config: GCodeConfig
) -> np.ndarray:
    """
    Per-point extrusion feedrate, slow at the build plate and ramping back
    to the nominal print speed.

    Point i sits at layer coordinate m = i / pointsPerLoop + 1 (windings
    above the plate, the first slice being one step up - the same coordinate
    adhesion.rampFirstLayers squashes against). The whole of the first
    winding (m in [1, 2]) prints at config.firstLayerFeedrate, then the
    speed ramps linearly and reaches config.printFeedrate at winding
    config.speedRampLayers:

        F(m) = firstLayerFeedrate + (printFeedrate - firstLayerFeedrate)
               * clip((m - 2) / (speedRampLayers - 2), 0, 1)

    A speedRampLayers of 2 or less means only the first winding is slowed,
    with no ramp after it. A firstLayerFeedrate at or above printFeedrate
    disables the ramp entirely.
    """
    if config.firstLayerFeedrate >= config.printFeedrate:
        return np.full(numPoints, config.printFeedrate, dtype=float)

    m = np.arange(numPoints, dtype=float) / pointsPerLoop + 1.0

    span = config.speedRampLayers - 2.0
    if span <= 0.0:
        frac = (m > 2.0).astype(float)
    else:
        frac = np.clip((m - 2.0) / span, 0.0, 1.0)

    return config.firstLayerFeedrate + (config.printFeedrate - config.firstLayerFeedrate) * frac


def emitGcode(
        flowSpiral: FlowSpiral,
        config: GCodeConfig,
        purge: PurgeConfig | None = None,
) -> GCodeProgram:
    """
    Turn a flow-compensated spiral path into a gcode program.

    Each segment i -> i+1 gets one G1 move on X, Y, Z and I, J, K. I/J/K carry
    the unit tangent vector of the centerline at that point, orienting the
    nozzle to stay perpendicular to the wall (see Spiral.calcTangentIJK).
    Each move carries the tangent of its endpoint, so orientation
    interpolates along with position.

    The extruded volume per segment is

        lineWidth * h * segmentLength * flow

    where flow is the per-point extrusion multiplier averaged over the
    segment's endpoints (compensates winding spacing on bends, see
    flow.calcFlow). Dividing by the filament cross-section area gives the
    E-axis distance. E is absolute, zeroed at the start of the body.

    Feedrate is modal: an F word is emitted only where the speed changes,
    which is the first-layer speed ramp near the build plate (see
    calcFeedrates) and nowhere else.

    Passing a PurgeConfig appends purge arcs to the preamble (see
    emitPurgeArcs); leaving it None emits no purge at all.
    """
    points = flowSpiral.spiral.points
    tangents = flowSpiral.spiral.tangents

    # B angle that aims the nozzle along the centerline tangent
    bRotation = -1 * calcBAngle(tangents)

    slicedPoints = np.column_stack((points[:, 0], points[:, 1], points[:, 2], bRotation))


    ## calculate flow
    flow = flowSpiral.flow

    segVectors = np.diff(points, axis=0)
    segLengths = np.linalg.norm(segVectors, axis=1)
    segFlow = (flow[:-1] + flow[1:]) / 2.0

    extrusionPerSeg = (
        config.lineWidth * flowSpiral.h * segLengths * segFlow
        / config.filamentArea()
    )
    extrusionTotals = np.cumsum(extrusionPerSeg)

    ## calculate feedrates (slow at the plate, ramping up)
    feedrates = calcFeedrates(points.shape[0], flowSpiral.spiral.pointsPerLoop, config)

    body = [
        "; --- spiral body ---",
        "M82 ; absolute extrusion",
        "G92 E0",
        f"G0 F{config.travelFeedrate:.0f} "
        f"X{slicedPoints[0, 0]:.3f} Y{slicedPoints[0, 1]:.3f} Z{slicedPoints[0, 2]:.3f} B{slicedPoints[0, 3]:.3f} \n"
        f"G1 F{feedrates[0]:.0f}",
    ]

    lastFeedrate = f"{feedrates[0]:.0f}"
    for i in range(segLengths.shape[0]):
        # F is modal; only write it where the ramp actually changes the speed
        feedrate = f"{feedrates[i + 1]:.0f}"
        feedWord = "" if feedrate == lastFeedrate else f"F{feedrate} "
        lastFeedrate = feedrate

        body.append(
            f"G1 {feedWord}"
            f"X{slicedPoints[i + 1, 0]:.3f} Y{slicedPoints[i + 1, 1]:.3f} Z{slicedPoints[i + 1, 2]:.3f} B{slicedPoints[i + 1, 3]:.3f} "
            f"E{extrusionTotals[i]:.5f}"
        )

    preamble = emitStartGcode(config)
    if purge is not None:
        preamble = preamble + emitPurgeArcs(config, purge)

    return GCodeProgram(
        preamble=preamble,
        body=body,
        postamble=emitEndGcode(config),
    )


def emitPurgeArcs(config: GCodeConfig, purge: PurgeConfig) -> list[str]:
    """
    Purge the nozzle before the part, the polar equivalent of the prime
    lines a cartesian slicer draws along the front edge of the bed.

    A straight prime line has no meaning on a polar bed, so each pass is
    instead an arc swept at a constant radius: the bed simply rotates under
    a parked nozzle. Pass p runs at

        r_p = purge.radius - p * purge.lineWidth

    sweeping purge.sweep degrees, alternating direction so the passes chain
    end to end without a travel between them. Arcs are discretized into G1
    segments every purge.segmentAngle degrees rather than emitted as G2/G3,
    since the machine's polar kinematics do the arc interpolation in the
    linear axes anyway and the tool path elsewhere is already linearized.

    Extrusion per segment matches the body's model,

        lineWidth * layerHeight * segmentLength / filamentArea

    with a stationary prime of purge.primeAmount at the start, a retract of
    purge.retract at the end, and an un-extruded wipe continuing the last
    arc so the bead breaks off away from the nozzle. E is absolute and
    zeroed here; emitGcode's body re-zeros it, so the purge cannot leak
    into the part's extrusion totals.

    Returned as a standalone block of lines - append it to a preamble, or
    pass a PurgeConfig to emitGcode to have it appended for you.
    """
    if purge.passes < 1:
        return ["; --- purge arcs (disabled: no passes) ---"]

    numSegments = max(1, int(np.ceil(abs(purge.sweep) / purge.segmentAngle)))
    volumePerLength = purge.lineWidth * purge.layerHeight / config.filamentArea()

    lines = [
        f"; --- purge arcs ({purge.passes} pass(es) at r={purge.radius:.1f}mm) ---",
        "M82 ; absolute extrusion",
        "G92 E0",
    ]

    extrusion = 0.0
    angle = purge.startAngle
    radius = purge.radius
    direction = 1.0

    for p in range(purge.passes):
        radius = purge.radius - p * purge.lineWidth
        if radius <= 0.0:
            lines.append(f"; pass {p} skipped: radius stepped through the bed centre")
            break

        # park at the start of the arc; on later passes this is a short
        # radial hop inward from where the previous pass ended
        start = np.radians(angle)
        lines.append(
            f"G0 F{config.travelFeedrate:.0f} "
            f"X{radius * np.cos(start):.3f} Y{radius * np.sin(start):.3f} "
            f"Z{purge.layerHeight:.3f} B0.000"
        )

        if p == 0 and purge.primeAmount > 0.0:
            extrusion += purge.primeAmount
            lines.append(f"G1 F{purge.feedrate:.0f} E{extrusion:.5f} ; prime")

        segAngle = direction * purge.sweep / numSegments
        segLength = 2.0 * radius * np.sin(abs(np.radians(segAngle)) / 2.0)
        extrusionPerSeg = volumePerLength * segLength

        lines.append(f"G1 F{purge.feedrate:.0f}")
        for i in range(1, numSegments + 1):
            theta = np.radians(angle + i * segAngle)
            extrusion += extrusionPerSeg
            lines.append(
                f"G1 X{radius * np.cos(theta):.3f} Y{radius * np.sin(theta):.3f} "
                f"E{extrusion:.5f}"
            )

        angle += direction * purge.sweep
        direction = -direction

    # wipe: carry on around the last arc without extruding, then retract
    if purge.wipeAngle > 0.0:
        theta = np.radians(angle - direction * purge.wipeAngle)
        lines.append(
            f"G1 F{config.travelFeedrate:.0f} "
            f"X{radius * np.cos(theta):.3f} Y{radius * np.sin(theta):.3f} ; wipe"
        )

    if purge.retract > 0.0:
        extrusion -= purge.retract
        lines.append(f"G1 F{purge.feedrate:.0f} E{extrusion:.5f} ; retract")

    lines.append("G92 E0")
    return lines


def emitStartGcode(config: GCodeConfig) -> list[str]:
    """
    Machine-specific startup: heat nozzle/bed, home axes, prime the nozzle.
    Skeleton only for now.
    """
    return ["; --- start gcode (not implemented) ---",
            f"M109 S{config.nozzleTemp}"]


def emitEndGcode(config: GCodeConfig) -> list[str]:
    """
    Machine-specific shutdown: retract, lift away from the part, park,
    turn off heaters and motors. Skeleton only for now.
    """
    return ["; --- end gcode (not implemented) ---"]
