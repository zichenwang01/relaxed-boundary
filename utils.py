import os
import sys
from os.path import join

import numpy as np 
import matplotlib.pyplot as plt

import drjit as dr 
import mitsuba as mi
mi.set_variant("cuda_ad_rgb") # choose backend
# mi.set_variant("llvm_ad_rgb") 

from mitsuba import Bool, Mask, UInt, Int, Float
from mitsuba import Point2i, Point2f, Vector2i, Vector2f, Ray2f
from mitsuba import Point3i, Point3f, Vector3i, Vector3f, Ray3f
from mitsuba import Color3f, TensorXf, Interaction3f, SurfaceInteraction3f

# ---------------------- CONSTANTS ----------------------
# main directory
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# data directory
DATA_DIR = join(MAIN_DIR, "data")

# config directory
CONFIG_DIR = join(MAIN_DIR, "configs")

# output directory for images
OUT_DIR = join(MAIN_DIR, "outputs")

# optimize directory
OPT_DIR = join(MAIN_DIR, "optimize")

# constant colors
white = Color3f(1, 1, 1)
black = Color3f(0, 0, 0)
red   = Color3f(1, 0, 0)
green = Color3f(0, 1, 0)
blue  = Color3f(0, 0, 1)

# ----------------------------- GLOBAL PARAMETER -------------------------------

# sphere tracing intersection threshold (at most 1e-7)
ITX_EPS = 1e-5

# ray marching step size at intersection (> ITX_EPS)
RAY_EPS = 1e-3

# hide environment lighting 
HIDE_ENV = Mask(False)
# HIDE_ENV = Mask(True)

# background color
BACKGROUND = black

# sdf interpolation method
SDF_MODE = 'linear'
# SDF_MODE = 'cubic'

# sdf threshold for boundary path 
SDF_EPS = 1e-4

# sdf direction derivative threshold for boundary path
SDF_DERIV_EPS = 1e-1

# itx refinement interval
NUM_REFINE = 0
ITX_REFINE = 1e-5

# graze point method
GRZ_MODE = 'bisection'

# get grz number of refinement
NUM_GRZ = 12

# ----------------------------------- LOSS ------------------------------------

def l2_loss(img, ref_img):
    """L2 loss"""
    return dr.mean(dr.sqr(img - ref_img))

def l1_loss(img, ref_img):
    """L1 loss"""
    return dr.mean(dr.abs(img - ref_img))

def laplacian_loss(data, _=None):
    """ Laplacian regularization loss
        Code from https://github.com/rgl-epfl/differentiable-sdf-rendering """

    def linear_idx(p):
        p.x = dr.clamp(p.x, 0, data.shape[0] - 1)
        p.y = dr.clamp(p.y, 0, data.shape[1] - 1)
        p.z = dr.clamp(p.z, 0, data.shape[2] - 1)
        return p.z * data.shape[1] * data.shape[0] + p.y * data.shape[0] + p.x

    shape = data.shape
    z, y, x = dr.meshgrid(*[dr.arange(mi.Float, shape[i]) for i in range(3)], indexing='ij')
    p = mi.Point3i(x, y, z)
    c = dr.gather(mi.Float, data.array, linear_idx(p))
    vx0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(-1, 0, 0)))
    vx1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(1, 0, 0)))
    vy0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, -1, 0)))
    vy1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 1, 0)))
    vz0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 0, -1)))
    vz1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 0, 1)))
    laplacian = dr.sqr(c - (vx0 + vx1 + vy0 + vy1 + vz0 + vz1) / 6)
    return dr.sum(laplacian)

# ------------------------------------ SDF ------------------------------------

def atleast_4d(tensor):
    """ Ensures a sensor has at least 4 dimensions 
        Code from https://github.com/rgl-epfl/differentiable-sdf-rendering """
    if tensor.ndim == 3:
        return tensor[..., None]
    return tensor

def redistance(phi, method='fmm'):
    """ Re-distances a signed distance field using the fast marching method
        Code adapted from https://github.com/rgl-epfl/differentiable-sdf-rendering"""
    if method == 'fastsweep': 
        # requires cuda
        import fastsweep
        return fastsweep.redistance(phi)
    elif method == 'fmm':
        import skfmm
        import numpy as np
        return skfmm.distance(phi, dx=1 / np.array(phi.shape))
    else:
        raise ValueError("Invalid re-distancing method")

def upsample_grid(data):
    """ Upsample a grid by a factor of 2 using linear interpolation"""
    return dr.upsample(mi.Texture3f(data, migrate=False), scale_factor=[2, 2, 2, 1]).tensor()

def upsample_sdf(sdf, grid):
    """ Upsample the sdf by doubling the resolution """
    res = grid.shape[0] * 2
    grid = dr.meshgrid(
        dr.linspace(Float, 0, 1, res),
        dr.linspace(Float, 0, 1, res),
        dr.linspace(Float, 0, 1, res)
    )
    grid = Point3f(grid[2], grid[0], grid[1])
    grid = sdf.at(grid)
    grid = TensorXf(grid, shape=(res, res, res, 1))
    return grid

def upsample_to(data, res):
    """ Upsample a grid to a specific resolution """
    cur_res = data.shape[0]
    while cur_res != res:
        data = upsample_grid(data)
        cur_res *= 2
    return data

def save_sdf(sdf:'SDF', dir:str, name:str):
    resolution = sdf.res
    grid = sdf.grid.tensor()
    grid = np.array(grid).flatten()
    grid = [resolution] + grid.tolist()
    np.savetxt(join(dir, name+".sdf"), grid, fmt='%f')

# --------------------------------- OPTIMIZE ----------------------------------

def get_sensors(num_sensor=12, resx=512, resy=512, radius=1.5):
    """Return a list of sensors arranged in a circle around the origin"""
    sensors = []
    for i in range(num_sensor):
        angle = 360.0 / num_sensor * i
        sensors.append(mi.load_dict({
            'type': 'perspective', 'fov': 45,
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm',
                'width': resx, 'height': resy,
                'filter': {'type': 'gaussian'}
            },
            'to_world': mi.ScalarTransform4f.translate([0.5, 0.5, 0.5]).rotate([0, 1, 0], angle).look_at(target=[0, 0, 0], origin=[0, 0, radius], up=[0, 1, 0]),
        }))
    return sensors

def get_h2_sensors(
    num_sensor_theta=12, num_sensor_phi=3, 
    resx=512, resy=512, radius=1.5, height=0.4
):
    """Return a list of sensors arranged evenly on a hemisphere"""
    sensors = []
    for j in range(num_sensor_phi):
        for i in range(num_sensor_theta // (j+1)):
            # Convert the index to spherical coordinates
            theta = np.pi * 2 * i / (num_sensor_theta // (j+1)) # azimuthal angle, from 0 to 2pi
            phi = np.pi / 3 * j / num_sensor_phi  # zenith angle, from 0 to pi/2

            # Convert spherical coordinates to Cartesian coordinates
            z = np.cos(phi) * np.cos(theta) * radius + 0.5
            x = np.cos(phi) * np.sin(theta) * radius + 0.5
            y = np.sin(phi) * radius + height

            sensors.append(mi.load_dict({
                'type': 'perspective', 'fov': 45,
                'sampler': {'type': 'independent'},
                'film': {
                    'type': 'hdrfilm',
                    'width': resx, 'height': resy,
                    'filter': {'type': 'gaussian'}
                },
                'to_world': mi.ScalarTransform4f.look_at(target=[0.5, 0.5, 0.5], origin=[x, y, z], up=[0, 1, 0]),
            }))
    return sensors

# --------------------------------- VIDEO ----------------------------------

def opt_video(frames_dir:str, video_dir:str, fps:int=12, view:int=0):
    """ Render a optimization video """
    
    # create a temporary text file
    np.savetxt('tmp.txt', np.array([]))
    
    # write image paths to a 
    with open('tmp.txt', 'w') as f:
        for i in range(5000):
            if os.path.exists(join(frames_dir, f'image-{i}-{view}.png')):
                f.write(f"file '{join(frames_dir, f'image-{i}-{view}.png')}'\n")

    # Use the text file as input for ffmpeg
    ffmpeg_cmd = f'ffmpeg -y -hide_banner -loglevel error -r {fps} -f concat -safe 0 -i tmp.txt -c:v libx264 -movflags +faststart -vf format=yuv420p -crf 15 -nostdin {video_dir}/optimize-{view}.mp4'

    # render video
    import subprocess
    subprocess.call(ffmpeg_cmd, shell=True)
    
    # delete temporary text file
    # os.remove('tmp.txt')

def run_ffmpeg(frame_name, video_path):
    """ Converts a sequence of frames to a video. 
        Requires ffmpeg to be installed on the system. 
        Code from https://github.com/rgl-epfl/differentiable-sdf-rendering """
    import shutil
    if shutil.which('ffmpeg') is None:
        print("Cannot find ffmpeg, skipping video generation")
        return

    frame_name = frame_name.replace(' ', '\\ ')
    video_path = video_path.replace(' ', '\\ ')
    ffmpeg_cmd = f'ffmpeg -y -hide_banner -loglevel error -i {frame_name} -c:v libx264 -movflags +faststart -vf format=yuv420p -crf 15 -nostdin {video_path}'
    
    import subprocess
    subprocess.call(ffmpeg_cmd, shell=True)

def rotate_video(
    video_dir:str, scene:mi.Scene, params, sdf:'SDF', 
    num_frame=128, resx=512, resy=512, spp=256
):
    """ Render a turntable video
        Code from https://github.com/rgl-epfl/differentiable-sdf-rendering """
    
    # create folder
    frame_dir = join(video_dir, "rotate_frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    # load integrator
    from integrators.direct import SDFDirectIntegrator
    integrator = mi.load_dict({"type": "sdf-direct"})
    
    for frame in range(num_frame):
        # camera parameters
        radius = 1.5
        angle = frame / num_frame * 2 * dr.pi
        o = mi.ScalarPoint3f(dr.cos(angle) * radius + 0.5, 0.5, dr.sin(angle) * radius + 0.5)
        
        # load camera
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 45.0,
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm', 
                'width': resx, 'height': resy, 
                'pixel_filter': {'type': 'gaussian'}
            }, 
            'to_world': mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(o[0], o[1], o[2]), [0.5, 0.5, 0.5], [0, 1, 0])
        })   
        
        # render
        with dr.suspend_grad():
            image = integrator.render(scene=scene, sdf=sdf, sensor=sensor, spp=spp, params=params)
            bitmap = mi.util.convert_to_bitmap(image)
            
        # save image
        mi.util.write_bitmap(join(frame_dir, f'frame{frame}.png'), bitmap)
    
    # convert frames to video
    frame_name = join(frame_dir, 'frame%d.png')
    video_name = join(video_dir, 'rotate.mp4')
    run_ffmpeg(frame_name, video_name)
    
def geometry_video(
    video_dir:str, scene:mi.Scene, sdf:'SDF', 
    num_frame=128, resx=512, resy=512, spp=256
):
    """ Render a turntable video of the geometry
        Code from https://github.com/rgl-epfl/differentiable-sdf-rendering """
    
    # create folder
    frame_dir = join(video_dir, "geo_frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    # load integrator
    from integrators.direct import SDFDirectIntegrator
    integrator = mi.load_dict({"type": "sdf-direct"})
    
    for frame in range(num_frame):
        # camera parameters
        radius = 1.5
        angle = frame / num_frame * 2 * dr.pi
        o = mi.ScalarPoint3f(dr.cos(angle) * radius + 0.5, 0.5, dr.sin(angle) * radius + 0.5)
        
        # load camera
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 45.0,
            'sampler': {'type': 'independent'},
            'film': {
                'type': 'hdrfilm', 
                'width': resx, 'height': resy, 
                'pixel_filter': {'type': 'gaussian'}
            }, 
            'to_world': mi.ScalarTransform4f.look_at(mi.ScalarPoint3f(o[0], o[1], o[2]), [0.5, 0.5, 0.5], [0, 1, 0])
        })   
        
        # render
        with dr.suspend_grad():
            image = integrator.render(scene=scene, sdf=sdf, sensor=sensor, spp=spp)
            bitmap = mi.util.convert_to_bitmap(image)
            
        # save image
        mi.util.write_bitmap(join(frame_dir, f'frame{frame}.png'), bitmap)
    
    # convert frames to video
    frame_name = join(frame_dir, 'frame%d.png')
    video_name = join(video_dir, 'geometry.mp4')
    run_ffmpeg(frame_name, video_name)