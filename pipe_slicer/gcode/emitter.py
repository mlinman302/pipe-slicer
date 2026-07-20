from pipe_slicer.types import FlowSpiral, GCodeConfig, GCodeProgram
import numpy as np


def emitGcode(flowSpiral: FlowSpiral, config: GCodeConfig) -> GCodeProgram:
    """
    Turn a flow-compensated spiral path into a gcode program.

    Each segment i -> i+1 gets one G1 move. The extruded volume per segment is

        lineWidth * h * segmentLength * flow

    where flow is the per-point extrusion multiplier averaged over the
    segment's endpoints (compensates winding spacing on bends, see
    flow.calcFlow). Dividing by the filament cross-section area gives the
    E-axis distance. E is absolute, zeroed at the start of the body.
    """
    points = flowSpiral.spiral.points
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
        f"X{points[0, 0]:.3f} Y{points[0, 1]:.3f} Z{points[0, 2]:.3f}",
        f"G1 F{config.printFeedrate:.0f}",
    ]
    body.extend(
        f"G1 X{points[i + 1, 0]:.3f} Y{points[i + 1, 1]:.3f} "
        f"Z{points[i + 1, 2]:.3f} E{extrusionTotals[i]:.5f}"
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
