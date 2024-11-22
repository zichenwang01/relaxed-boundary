import os 
import sys
import argparse
from os.path import join
this_dir = os.path.realpath(__file__)
this_dir = os.path.dirname(this_dir)
main_dir = os.path.dirname(this_dir)
sys.path.append(main_dir)

import numpy as np

import utils
from utils import *
from sdf import *
from integrators.direct import *

# ---------------------- SAMPLING ----------------------
def sample_mesh(mesh_scene, N=100):
    params = mi.traverse(mesh_scene)
    
    positions = np.empty((0, 3))
    normals = np.empty((0, 3))
    for key in params.keys():
        # Check if the key corresponds to a mesh's faces
        if key.endswith('.faces'):
            # read faces
            faces = np.array(params[key]).reshape(-1, 3)
            
            # read vertices
            vertices_key = key.replace('.faces', '.vertex_positions')
            vertices = np.array(params[vertices_key]).reshape(-1, 3)
           
            for _ in range(N):
                # Randomly select a face
                face_index = np.random.randint(faces.shape[0])
                face = faces[face_index]

                # Get the vertices of the face
                v1, v2, v3 = vertices[face]

                # Generate random Barycentric coordinates
                u = np.random.random()
                v = np.random.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                w = 1 - u - v

                # Compute the position of the random point
                point = u * v1 + v * v2 + w * v3

                # Compute the normal at the point
                normal = np.cross(v2 - v1, v3 - v1)
                try:
                    normal = normal / np.linalg.norm(normal)
                except:
                    print(point)
                    print(np.linalg.norm(normal))
                    continue

                positions = np.concatenate((positions, [point]))
                normals = np.concatenate((normals, [normal]))

    return Point3f(positions), Vector3f(normals)

# ---------------------- EVAL 3D ----------------------
def eval_3d(exp_dir, sdf_dir, mesh_dir):
    print("----- start eval 3d -----")
    
    # Create eval directory
    eval_dir = join(OUT_DIR, exp_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Load SDF
    sdf = SDF()
    sdf.load(sdf_dir)

    # Load mesh scene
    mesh_scene = mi.load_file(
        './data/eval/diffuse-scene.xml',
        shape_file=mesh_dir
    )
    print("----- finish load scene -----")

    # Sample mesh
    positions, normals = sample_mesh(mesh_scene)

    # Chamfer distance
    distance = dr.abs(sdf.at(positions))
    distance_error = dr.mean(distance)
    print("Distance error:", distance_error)

    # Save evaluation
    with open(join(eval_dir, 'eval_3d.txt'), 'w') as f:
        f.write(f"Distance error: {distance_error}\n")
    
    print("----- finish eval 3d -----")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp_dir', 
        type=str, required=True, 
        help='Experiment directory'
    )
    parser.add_argument('--scene',
        type=str, required=True,
        help='Scene'
    )
    args = parser.parse_args()
    
    exp_dir = join(OUT_DIR, args.exp_dir)
    
    mesh_dir = join(DATA_DIR, args.scene, f"{args.scene}-shape.xml")
    
    sdf_dir = join(exp_dir, 'params', 'sdf_final.npy')

    eval_3d(exp_dir, sdf_dir, mesh_dir)