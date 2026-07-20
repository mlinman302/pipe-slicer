"""
Basic runner to iteratively test
"""
from pipe_slicer.io import load
from pipe_slicer.algorithm import centerline, slicer, spiral
from pipe_slicer.flow import flow
from pipe_slicer.gcode import emitter
from pathlib import Path
from pipe_slicer.types import GCodeConfig
import trimesh

TEST_DATA = Path.cwd() / "tests" / "tubes"



### import and segmentation ###
print(Path.cwd())
try:
    importTube = load.importTube(TEST_DATA / "tube.STL")
except TypeError:
    print("Error in opening test tube file")
segmented = load.segmentMesh(importTube)

### centerline analysis ###
cl = centerline.calcCenterline(segmented, wall="outer")

### slicing and flow ###
step_dist = 0.4
slices = slicer.calcSlices(cl, segmented.outer, step_dist=step_dist)
spiral = spiral.calcSpiralPath(slices)

flowSpiral = flow.calcFlow(spiral, minFlow=0.1, maxFlow=10, collisionRatio=0.2, gapRatio=4, h=step_dist)
print(flowSpiral.chars.summary())


### gcode emission ###
emitter.emitGcode(flowSpiral, GCodeConfig()).save(TEST_DATA / "gcodeOut" / "tube.gcode")













