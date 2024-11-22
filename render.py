from utils import *
from sdf import *

from integrators.albedo import AlbedoIntegrator
from integrators.normal import SDFNormalIntegrator
from integrators.sample import SDFDirectSampleIntegrator
from integrators.direct import DirectIntegrator, SDFDirectIntegrator
from integrators.direct_diff import SDFDirectDiffIntegrator
from integrators.path import PathMISIntegrator, SDFPathIntegrator
from integrators.path_diff import SDFPathDiffIntegrator

# ---------------------- LOAD SCENE ---------------------- 
# scene name
scene_name = "vbunny"
SCENE_DIR = join(DATA_DIR, scene_name)

# load scene
scene : mi.Scene = mi.load_file(join(SCENE_DIR, "scene.xml"))

# # load scene with textures
# texure_dir = '/home/zw336/IR/bp2/exp/6.teaser/res=64_lr=0.005_decay=2.5_scale=5_spp=128_sensors=24+4_batch=5_eps=1e-05_up=[1000,2000,3000]/sdf/final.vol'
# scene : mi.Scene = mi.load_file(
#     '/home/zw336/IR/bp2/data/textures/scene.xml',
#     texture = texure_dir
# )

# load scene sensor
sensor : mi.Sensor = scene.sensors()[0]

# set spp
spp = 1024

# ----------------------- REF IMAGE ----------------------- 
print("----- render ref image -----")

# load integrator
mesh_integrator = mi.load_dict({'type': 'direct'})

with dr.suspend_grad():
    # render ref image
    image = mi.render(scene=scene, sensor=sensor, spp=spp)
    bitmap = mi.util.convert_to_bitmap(image)
    
    # # render custom ref image
    # sdf = SDF()
    # image = mesh_integrator.render(scene=scene, sdf=sdf, sensor=sensor, spp=spp, max_depth = 3)
    # bitmap = mi.util.convert_to_bitmap(image)
    
    # save ref image
    mi.util.write_bitmap(join(OUT_DIR, "image_ref.png"), bitmap)
    mi.util.write_bitmap(join(OUT_DIR, "image_ref.exr"), image)
    
print("----- finish render ref image -----")

# ----------------------- LOAD SDF ------------------------ 
print("----- load sdf -----")

# sdf resolution
sdf_res = 256
# sdf_res = 512

# load sdf
sdf = SDF()
sdf.load('/home/zw336/IR/relaxed-boundary-private/outputs/2024-10-19_12-21-54/params/sdf_final.npy')

print("----- finish load sdf -----")

# ----------------------------- SDF IMAGE ---------------------------- 
print("----- render sdf image -----")

# load integrator
sdf_integrator = mi.load_dict({'type': 'sdf-direct'})
# sdf_integrator = mi.load_dict({'type': 'sdf-direct-diff'})

# render sdf image
dr.kernel_history_clear()
image = sdf_integrator.render(scene=scene, sdf=sdf, sensor=sensor, spp=spp)
bitmap = mi.util.convert_to_bitmap(image)
history = dr.kernel_history()
total_time = sum(h['execution_time'] for h in history)

# print time
print(f"Total time: {total_time:.2f}s")

# save sdf image
mi.util.write_bitmap(join(OUT_DIR, "image_sdf.png"), bitmap)
mi.util.write_bitmap(join(OUT_DIR, "image_sdf.exr"), image)

print("----- finish render sdf image -----")