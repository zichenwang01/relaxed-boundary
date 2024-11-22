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

# ---------------------- MEASURES ----------------------
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max(np.max(img1), np.max(img2))
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

# ---------------------- EVAL 2D ----------------------
def eval_2d(
    exp_dir, sdf_dir, mesh_dir, 
    albedo_dir=None, roughness_dir=None, metallic_dir=None
):
    print("----- start 2D evaluation -----")
    
    # Create eval directory
    eval_dir = join(OUT_DIR, exp_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Load sensors
    num_sensor = 12
    sensors = get_sensors(num_sensor=num_sensor, resx=512, resy=512)

    # Load integrators
    mesh_integrator = DirectIntegrator()
    sdf_integrator = SDFDirectIntegrator()

    # Load SDF
    sdf = SDF()
    sdf.load(sdf_dir)

    # set render background
    utils.HIDE_ENV = Mask(True)
    utils.background = white

    for envmap in ['studio', 'nature', 'indoor']:
        print(f"----- render {envmap} image -----")

        # Create envmap directory
        envmap_dir = join(eval_dir, '2d', envmap)
        os.makedirs(envmap_dir, exist_ok=True)

        psnr_val = 0.0
        ssim_val = 0.0
        lpips_val = 0.0
        
        for view in range(num_sensor):
            # Set angle
            angle = 0 if envmap == 'studio' else 30 * view
            
            # Load mesh scene
            mesh_scene = mi.load_file(
                './data/eval/principled-scene.xml',
                envmap=envmap,
                angle=angle,
                shape_file=mesh_dir
            )
            
            # Load sdf scene
            if albedo_dir is not None:
                if roughness_dir is not None:
                    if metallic_dir is not None:
                        sdf_scene = mi.load_file(
                            './data/eval/metallic-scene.xml',
                            envmap=envmap,
                            angle=angle,
                            albedo=albedo_dir,
                            roughness=roughness_dir,
                            metallic=metallic_dir
                        )
                    else:
                        sdf_scene = mi.load_file(
                            './data/eval/principled-scene.xml',
                            envmap=envmap,
                            angle=angle,
                            albedo=albedo_dir,
                            roughness=roughness_dir,
                        )
                else:
                    sdf_scene = mi.load_file(
                        './data/eval/diffuse-scene.xml',
                        envmap=envmap,
                        angle=angle,
                        albedo=albedo_dir,
                    )
            else:
                sdf_scene = mi.load_file(
                    './data/eval/diffuse-scene.xml',
                    envmap=envmap,
                    angle=angle,
                )
        
            # Render mesh image
            mesh_image = mesh_integrator.render(scene=mesh_scene, sdf=sdf,
                                                sensor=sensors[view], spp=1023)
            mesh_bitmap = mi.util.convert_to_bitmap(mesh_image)
            mi.util.write_bitmap(join(envmap_dir, f"mesh_{view}.png"), mesh_bitmap)
            
            # Render sdf image
            sdf_image = sdf_integrator.render(scene=sdf_scene, sdf=sdf,
                                              sensor=sensors[view], spp=1023)
            sdf_bitmap = mi.util.convert_to_bitmap(sdf_image)
            mi.util.write_bitmap(join(envmap_dir, f"sdf_{view}.png"), sdf_bitmap)
            
            # Change to numpy
            image1 = np.array(mesh_image)
            image2 = np.array(sdf_image)
            
            # Evaluate
            psnr_val  += psnr(image1, image2)
            ssim_val  += ssim(image1, image2)
            lpips_val += lpips(image1, image2)
        
        psnr_val  /= float(num_sensor)
        ssim_val  /= float(num_sensor)
        lpips_val /= float(num_sensor)

        # Save evaluation
        with open(join(eval_dir, f"eval_2d.txt"), 'a') as f:
            f.write(f"Envmap: {envmap}\n")
            f.write(f"PSNR: {psnr_val}\n")
            f.write(f"SSIM: {ssim_val}\n")
            f.write(f"LPIPS: {lpips_val}\n")

        print(f"----- finish render {envmap} image -----")
    print("----- finish 2D evaluation -----")
        
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