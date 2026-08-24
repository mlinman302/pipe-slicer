import numpy as np
import trimesh as tm
import trimesh.visual as visual
from pathlib import Path
import networkx as nx

from pipe_slicer.types import SegmentedMesh

colors = [
    [220,  50,  50, 255],  # red, starting face
    [ 50, 180,  80, 255],  # green, terminating face
    [ 70, 130, 180, 255],  # blue, inner wall
    [230, 170,  30, 255],  # orange, outer wall
]

tm.util.attach_to_log() # enable trimesh print messages

def importTube(path: str | Path) -> tm.Trimesh:
    mesh_import = tm.load_mesh(str(path), force="mesh")

    if not isinstance(mesh_import, tm.Trimesh) or len(mesh_import.faces) == 0:
        # failed import
        raise TypeError(f"{path}: import failed to yield mesh")

    mesh = mesh_import

    ### standard trimesh cleanup ###
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if len(mesh.faces) == 0:
        # failed after cleanup
        raise TypeError(f"{path}: zero faces after cleanup")
    
    return mesh
    

def segmentMesh(mesh: tm.Trimesh) -> SegmentedMesh:


    # traverse mesh as graph. if face angle is above threshold angle, do not add to network
    lower_angle_threshold = np.radians(85)
    upper_angle_threshold = np.radians(95)


    # really just splitting out ~90 degree angles
    lower_smooth = mesh.face_adjacency_angles < lower_angle_threshold
    upper_smooth = mesh.face_adjacency_angles > upper_angle_threshold
    smooth = lower_smooth | upper_smooth

    face_graph = nx.Graph()
    face_graph.add_nodes_from(range(len(mesh.faces))) # add all faces incides as nodes
    face_graph.add_edges_from(mesh.face_adjacency[smooth]) # add adjacency edges where smooth

    faces = list(nx.connected_components(face_graph))


    meshes: list[tm.Trimesh] = []
    for face in faces:
        face_indices = np.array(list(face))
        meshed_face = mesh.submesh([face_indices], append=True) # ensures output is only Trimesh, not List[Trimesh]
        assert isinstance(meshed_face, tm.Trimesh) # will pass dt above statement
        meshes.append(meshed_face)


    if len(meshes) != 4:
        raise TypeError("Not four faces in mesh. Error in splitting or multi-branched tube")

    #sort by largest area
    meshes.sort(key=lambda face:face.area)

    # sort smallest two faces (end faces) by location: closest to origin is the start face
    meshes[0:2] = sorted(meshes[0:2], key=lambda m: np.dot(m.centroid, m.centroid))


    # apply colors to different meshes
    for mesh, color in zip(meshes, colors):
        mesh.visual = visual.ColorVisuals(
        mesh, face_colors=np.tile(color, (len(mesh.faces), 1))
    )

    

    sm =  SegmentedMesh(*meshes)

    num_bottom_facets = len(sm.bottom.facets)
    if num_bottom_facets > 1:
        raise ValueError(
            f"Number of facets {num_bottom_facets} not equal to one. Bottom surface must be planar.")


    return sm

