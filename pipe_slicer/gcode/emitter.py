from pipe_slicer.types import FlowSpiral, GCodeConfig, GCodeProgram
import numpy as np
import matplotlib.pyplot as plt

def rotOntoXZ(vectors: np.ndarray) -> np.ndarray:
    """Rotate row vectors about onto XZ plane"""
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    theta = np.arctan2(y, z)

    x_ = x
    y_ = y * np.cos(theta) - z * np.sin(theta)
    z_ = y * np.sin(theta) + z * np.cos(theta)

    return np.column_stack((x_, y_, z_))


def emitGcode(flowSpiral: FlowSpiral, config: GCodeConfig) -> GCodeProgram:
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
    """
    points = flowSpiral.spiral.points
    tangents = flowSpiral.spiral.tangents

    # rotate the tangent vector onto the XZ plane
    toolheadVector = rotOntoXZ(tangents)

    # determine the angle of the vector relative to + z
    cosBRotation = np.dot(toolheadVector, [0, 0, 1])

    # use that to determine B angle
    bRotation = np.degrees(np.arccos(cosBRotation))

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

    body = [
        "; --- spiral body ---",
        "M82 ; absolute extrusion",
        "G92 E0",
        f"G0 F{config.travelFeedrate:.0f} "
        f"X{slicedPoints[0, 0]:.3f} Y{slicedPoints[0, 1]:.3f} Z{slicedPoints[0, 2]:.3f} B{slicedPoints[0, 3]:.3f} \n"
        f"G1 F{config.printFeedrate:.0f}",
    ]
    body.extend(
        f"G1 X{slicedPoints[i + 1, 0]:.3f} Y{slicedPoints[i + 1, 1]:.3f} Z{slicedPoints[i + 1, 2]:.3f} B{slicedPoints[i + 1, 3]:.3f} "
        # f"E{extrusionTotals[i]:.5f}"
        for i in range(segLengths.shape[0])
    )

    return GCodeProgram(
        preamble=emitStartGcode(config),
        body=body,
        postamble=emitEndGcode(config),
    )


def emitStartGcode(config: GCodeConfig) -> list[str]:
    """
    Machine-specific startup: heat nozzle/bed, home axes, prime the nozzle.
    Skeleton only for now.
    """
    return ["; --- start gcode (not implemented) ---"]


def emitEndGcode(config: GCodeConfig) -> list[str]:
    """
    Machine-specific shutdown: retract, lift away from the part, park,
    turn off heaters and motors. Skeleton only for now.
    """
    return ["; --- end gcode (not implemented) ---"]
