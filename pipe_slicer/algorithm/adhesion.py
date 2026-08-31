from pipe_slicer.types import FlowSpiral, Spiral
import dataclasses
import numpy as np


def rampFirstLayers(
        flowSpiral: FlowSpiral,
        firstLayerRatio: float = 0.75,
        rampLayers: float = 3.0
) -> FlowSpiral:
    """
    Squash the first few windings towards the build plate so the print
    sticks better, then ramp the winding spacing back to nominal.

    Slices start one step (h) above the bottom face, so the winding at
    layer coordinate m (m = 1 for the first winding, counted in windings
    above the plate) nominally sits at height

        H(m) = h * m

    measured along the centerline. The ramp replaces the constant winding
    spacing with a profile that starts squashed and recovers linearly:

        s(u) = firstLayerRatio + (1 - firstLayerRatio) * min(u / rampLayers, 1)

    so the squashed height is H'(m) = h * integral of s from 0 to m, and the
    displacement applied to each path point is

        offset(m) = H(m) - H'(m)
                  = h * (1 - r) * (m - m^2 / (2 * L))   for m <= L
                  = h * (1 - r) * L / 2                 for m >  L

    which is zero at the plate, grows through the ramp, and is constant
    above it - i.e. everything past the ramp is rigidly shifted down, and
    only the ramped windings actually change spacing.

    The offset is applied along -tangent (the direction the spiral advances
    along the centerline), not along -Z, so a tube whose start is tilted off
    the bed normal is squashed towards its own bottom face.

    Extrusion is deliberately left at nominal: `flow` and `spacing` still
    describe the un-ramped path, so the squashed windings receive a full
    layer's worth of filament in a thinner gap and spread wider against the
    plate. That extra bead width is the adhesion.

    Returns a new FlowSpiral; the input is left untouched.
    """
    if not 0.0 < firstLayerRatio <= 1.0:
        raise ValueError(f"firstLayerRatio must be in (0, 1], got {firstLayerRatio}")

    if rampLayers <= 0.0:
        raise ValueError(f"rampLayers must be positive, got {rampLayers}")

    if firstLayerRatio == 1.0:
        return flowSpiral

    spiral = flowSpiral.spiral
    h = flowSpiral.h

    numPoints = spiral.points.shape[0]

    # layer coordinate of every path point: windings above the build plate.
    # the first path point is the first slice, one step above the bottom face.
    m = np.arange(numPoints, dtype=float) / spiral.pointsPerLoop + 1.0

    squash = 1.0 - firstLayerRatio
    inRamp = m <= rampLayers

    offset = np.where(
        inRamp,
        h * squash * (m - m ** 2 / (2.0 * rampLayers)),
        h * squash * rampLayers / 2.0
    )

    points = spiral.points - offset[:, np.newaxis] * spiral.tangents

    ramped = Spiral(points, spiral.pointsPerLoop, spiral.tangents)

    return dataclasses.replace(flowSpiral, spiral=ramped)
