import os 
import sys
import argparse
from os.path import join
this_dir = os.path.realpath(__file__)
this_dir = os.path.dirname(this_dir)
main_dir = os.path.dirname(this_dir)
sys.path.append(main_dir)

from utils import *
from eval_2d import eval_2d
from eval_3d import eval_3d
from eval_inverse import eval_inverse

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
    
    albedo_dir = join(exp_dir, 'params', 'albedo_final.vol')
    if not os.path.exists(albedo_dir):
        albedo_dir = None
        
    roughness_dir = join(exp_dir, 'params', 'roughness_final.vol')
    if not os.path.exists(roughness_dir):
        roughness_dir = None
        
    metallic_dir = join(exp_dir, 'params', 'metallic_final.vol')
    if not os.path.exists(metallic_dir):
        metallic_dir = None

    eval_2d(
        exp_dir, sdf_dir, mesh_dir, 
        albedo_dir, roughness_dir, metallic_dir
    )
    
    eval_3d(exp_dir, sdf_dir, mesh_dir)
    
    eval_inverse(
        exp_dir, sdf_dir, mesh_dir, 
        albedo_dir, roughness_dir, metallic_dir
    )