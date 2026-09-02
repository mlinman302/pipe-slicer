"""
Basic runner to iteratively test
"""
from pipe_slicer.io import load
from pipe_slicer.algorithm import adhesion, centerline, slicer, spiral
from pipe_slicer.flow import flow
from pipe_slicer.gcode import emitter
from pathlib import Path
from pipe_slicer.types import GCodeConfig, PurgeConfig

TEST_DATA = Path.cwd() / "tests" / "tubes"



### import and segmentation ###
print(Path.cwd())
try:
    importTube = load.importTube(TEST_DATA / "tube.STL")
except TypeError:
    print("Error in opening test tube file")
segmented = load.alignToBuildPlate(load.segmentMesh(importTube))

### centerline analysis ###
cl = centerline.calcCenterline(segmented, wall="outer")

### slicing and flow ###
step_dist = 0.4
slices = slicer.calcSlices(cl, segmented.outer, step_dist=step_dist)
spiral = spiral.calcSpiralPath(slices)

flowSpiral = flow.calcFlow(spiral, minFlow=0.1, maxFlow=10, collisionRatio=0.2, gapRatio=4, h=step_dist)
print(flowSpiral.chars.summary())

### bed adhesion: squash the first windings and print them slow, ramping
### both back to nominal over the same number of windings ###
rampLayers = 3
flowSpiral = adhesion.rampFirstLayers(flowSpiral, firstLayerRatio=0.75, rampLayers=rampLayers)


### gcode emission ###
emitter.emitGcode(
    flowSpiral,
    GCodeConfig(lineWidth=1.2,  speedRampLayers=rampLayers),
    purge=PurgeConfig()
).save(TEST_DATA / "gcodeOut" / "tube.gcode")













