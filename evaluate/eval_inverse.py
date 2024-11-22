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
from integrators.normal import *
from integrators.albedo import *

# ---------------------- MEASURES ----------------------
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

from skimage import img_as_float
from skimage.metrics import structural_similarity
def ssim(img1, img2):
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)
    return structural_similarity(img1, img2, data_range=img2.max() - img2.min(), channel_axis=2)

import lpips, torch
loss_fn = lpips.LPIPS(net='alex')
def lpips(img1, img2):
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    return loss_fn(img1, img2).item()

# ---------------------- EVAL INVERSE ----------------------
def eval_inverse(
    exp_dir, sdf_dir, mesh_dir, 
    albedo_dir=None, roughness_dir=None, metallic_dir=None
):
    print("----- start eval inverse -----")
    
    # Create eval directory
    eval_dir = join(OUT_DIR, exp_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Create inverse directory
    inverse_dir = join(OUT_DIR, exp_dir, 'eval', 'inverse')
    os.makedirs(inverse_dir, exist_ok=True)

    # Load sensors
    num_sensor = 12
    sensors = get_sensors(num_sensor=num_sensor, resx=1024, resy=1024)

    # Load SDF
    sdf = SDF()
    sdf.load(sdf_dir)

    # set render background
    utils.HIDE_ENV = Mask(True)
    utils.background = white

    for view in range(num_sensor):
        # Set angle
        angle = 30 * view
        
        # Load mesh scene
        mesh_scene = mi.load_file(
            './data/eval/principled-scene.xml',
            envmap='studio',
            angle=angle,
            shape_file=mesh_dir
        )
        
        # Load sdf scene
        if albedo_dir is not None:
            if roughness_dir is not None:
                if metallic_dir is not None:
                    sdf_scene = mi.load_file(
                        './data/eval/metallic-scene.xml',
                        envmap='studio',
                        angle=angle,
                        albedo=albedo_dir,
                        roughness=roughness_dir,
                        metallic=metallic_dir
                    )
                else:
                    sdf_scene = mi.load_file(
                        './data/eval/principled-scene.xml',
                        envmap='studio',
                        angle=angle,
                        albedo=albedo_dir,
                        roughness=roughness_dir,
                    )
            else:
                sdf_scene = mi.load_file(
                    './data/eval/diffuse-scene.xml',
                    envmap='studio',
                    angle=angle,
                    albedo=albedo_dir,
                )
        else:
            sdf_scene = mi.load_file(
                './data/eval/diffuse-scene.xml',
                envmap='studio',
                angle=angle,
            )

        for measure in ['direct', 'normal', 'albedo']:
            # Load integrator
            mesh_integrator = mi.load_dict({'type': f'{measure}'})
            sdf_integrator = mi.load_dict({'type': f'sdf-{measure}'})
            
            # Render mesh image
            image1 = mesh_integrator.render(scene=mesh_scene, sdf=sdf, 
                                            sensor=sensors[view], spp=1023)
            bitmap1 = mi.util.convert_to_bitmap(image1)
            mi.util.write_bitmap(join(inverse_dir, f"mesh_{measure}_{view}.png"), bitmap1)
            
            # Render sdf image
            image2 = sdf_integrator.render(scene=sdf_scene, sdf=sdf, 
                                           sensor=sensors[view], spp=1023)
            bitmap2 = mi.util.convert_to_bitmap(image2)
            mi.util.write_bitmap(join(inverse_dir, f"sdf_{measure}_{view}.png"), bitmap2)
            
            # Change to numpy
            image1 = np.array(image1)
            image2 = np.array(image2)
            
            # Evaluate
            psnr_val = psnr(image1, image2)
            ssim_val = ssim(image1, image2)
            lpips_val = lpips(image1, image2)
            
            with open(join(inverse_dir, f"eval_{measure}.txt"), 'a') as f:
                f.write(f"View {view}:\n")
                f.write(f"PSNR: {psnr_val}\n")
                f.write(f"SSIM: {ssim_val}\n")
                f.write(f"LPIPS: {lpips_val}\n")
                
    print("----- finish eval inverse -----")

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

    eval_inverse(
        exp_dir, sdf_dir, mesh_dir, 
        albedo_dir, roughness_dir, metallic_dir
    )