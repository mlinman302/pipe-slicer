from dataclasses import dataclass
import numpy as np
import scipy.interpolate
import trimesh
import trimesh.path as path
import trimesh.path.entities as entities
import skeletor


@dataclass(frozen=True)
class SegmentedMesh:
    """
    Collection of segmented faces making a whole tube
    """
    bottom: trimesh.Trimesh # attached to build plate
    top: trimesh.Trimesh    # terminating face
    inner: trimesh.Trimesh  # inner wall (respected truely during slicing)
    outer: trimesh.Trimesh  # outer wall (not used for much)

    def toScene(self) -> trimesh.Scene:
        scene = trimesh.Scene()
        scene.add_geometry(self.bottom)
        scene.add_geometry(self.top)
        scene.add_geometry(self.inner)
        scene.add_geometry(self.outer)
        return scene

@dataclass(frozen=True)
class Centerline:
    spline: scipy.interpolate.BSpline # smooth spline representing centerline
    rawPath: np.ndarray               # list of points on the raw path


@dataclass(frozen=True)
class Frame:
    tangent: np.ndarray
    reference: np.ndarray
    origin: np.ndarray

    def calcS(self) -> np.ndarray:
        cross = np.cross(self.tangent, self.reference)
        return cross / np.linalg.norm(cross)

@dataclass(frozen=True)
class Slice:
    ring3D: np.ndarray
    zeroPointIdx: int
    frame: Frame # RMF frame the slice was cut at; defines the azimuth parametrization


    def toPath3D(self) -> path.path.Path3D:
    
        vertices = self.ring3D

        colorGray = [155, 155, 155]
        colorOrange = [255, 170, 0]
        if np.allclose(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        
        numPoints = vertices.shape[0]

        colors = [colorOrange if idx == self.zeroPointIdx else colorGray for idx in range(numPoints)]

        e = [
            entities.Line(points=[i, (i+1) % numPoints], color=colors[i])
            for i in range(numPoints)
        ]

        return path.path.Path3D(entities=e, vertices=vertices)
            
            
@dataclass(frozen=True)
class Spiral:
    points: np.ndarray
    pointsPerLoop: int # samples per full winding (theta advances 2*pi every pointsPerLoop points)

    def toPath3D(self):
        numPoints = self.points.shape[0]

        e = [
            entities.Line(points=[i, i + 1])
            for i in range(numPoints - 1)
        ]
        return path.path.Path3D(entities=e, vertices=self.points)



@dataclass(frozen=True)
class FlowCharacteristics:
    """
    Summary statistics of the per-point extrusion flow multipliers,
    for analysing how much a path deviates from nominal flow (1.0).
    """
    flowMin: float        # smallest flow multiplier (after clamping)
    flowMax: float        # largest flow multiplier (after clamping)
    flowMean: float       # mean flow multiplier
    stdDev: float         # standard deviation of the flow multipliers
    clampedFraction: float # fraction of points whose raw spacing/h fell outside [minFlow, maxFlow]

    def summary(self) -> str:
        return (
            f"flow in [{self.flowMin:.3f}, {self.flowMax:.3f}], "
            f"mean {self.flowMean:.3f} +/- {self.stdDev:.3f}, "
            f"{self.clampedFraction:.1%} of points clamped"
        )



@dataclass(frozen=True)
class GCodeConfig:
    """
    Printer and material parameters needed to turn a path into extrusion moves.
    """
    filamentDiameter: float = 1.75 # filament stock diameter
    lineWidth: float = 0.45        # extruded bead width (single wall in vase mode)
    printFeedrate: float = 1200.0  # extrusion move speed, mm/min
    travelFeedrate: float = 6000.0 # non-extruding move speed, mm/min
    nozzleTemp: float = 210.0
    bedTemp: float = 60.0

    def filamentArea(self) -> float:
        return np.pi * (self.filamentDiameter / 2.0) ** 2


@dataclass(frozen=True)
class GCodeProgram:
    """
    Emitted gcode split into its three phases, kept separate so the
    machine-specific preamble/postamble can be swapped without touching
    the tool path itself.
    """
    preamble: list[str]  # heating, homing, priming
    body: list[str]      # the spiral tool path
    postamble: list[str] # retract, park, shut down

    def toString(self) -> str:
        return "\n".join(self.preamble + self.body + self.postamble) + "\n"

    def save(self, filePath) -> None:
        with open(filePath, "w") as f:
            f.write(self.toString())


@dataclass(frozen=True)
class FlowSpiral:
    """
    Spiral path annotated with per-point winding spacing and the extrusion
    flow multiplier that compensates for it (inside of a bend squeezes
    windings together, outside stretches them apart).
    """
    spiral: Spiral
    h: float             # nominal winding spacing (slice step distance)
    spacing: np.ndarray  # per-point distance to the previous winding (h where no winding below)
    flow: np.ndarray     # per-point extrusion multiplier: spacing / h, clamped
    chars: FlowCharacteristics # characteristics of the flow (for analysis purposes)
    def toPath3D(self) -> path.path.Path3D:
        """Path3D with segments colored by flow: blue = starved, gray = nominal, red = overfed."""
        points = self.spiral.points
        numPoints = points.shape[0]

        e = []
        for i in range(numPoints - 1):
            # map flow 0.5 -> blue, 1.0 -> gray, 1.5 -> red
            deviation = np.clip((self.flow[i] - 1.0) / 0.5, -1.0, 1.0)
            if deviation >= 0:
                color = [155 + int(100 * deviation), 155 - int(100 * deviation), 155 - int(100 * deviation)]
            else:
                color = [155 + int(100 * deviation), 155 + int(100 * deviation), 155 - int(100 * deviation)]
            e.append(entities.Line(points=[i, i + 1], color=color))

        return path.path.Path3D(entities=e, vertices=points)