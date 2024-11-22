import os
import sys
import json
import time
import argparse
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils
from utils import *
from sdf import *
from integrators import *

def exact(args, config):
    return args if args is not None else config

def main(args):
    
    # ------------------------------ LOAD CONFIG ------------------------------
    config_file = join(CONFIG_DIR, f"{args.config}.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # update experiment parameters
    exp_config = config["experiment"]
    exp_config["exp_name"] = exact(exp_config["exp_name"], time.strftime("%Y-%m-%d_%H-%M-%S"))
    exp_config["exp_name"] = exact(args.name, exp_config["exp_name"])
    exp_config["scene_name"] = exact(args.scene, exp_config["scene_name"])
    exp_config["num_epoch"] = exact(args.num_epochs, exp_config["num_epoch"])
    exp_config["lr"] = exact(args.lr, exp_config["lr"])
    exp_config["num_sensor_theta"] = exact(args.num_sensor_theta, exp_config["num_sensor_theta"])
    exp_config["num_sensor_phi"] = exact(args.num_sensor_phi, exp_config["num_sensor_phi"])
    exp_config["batch_size"] = exact(args.batch_size, exp_config["batch_size"])
    
    # update rendering parameters
    render_config = config["rendering"]
    render_config["sdf_integrator"] = exact(args.integrator, render_config["sdf_integrator"])
    render_config["hide_env"] = exact(args.hide_env, render_config["hide_env"])
    render_config["resx"] = exact(args.resx, render_config["resx"])
    render_config["resy"] = exact(args.resy, render_config["resy"])
    render_config["spp"] = exact(args.spp, render_config["spp"])
    render_config["spp_grad"] = exact(args.spp_grad, render_config["spp_grad"])
    render_config["sdf_mode"] = exact(args.sdf_mode, render_config["sdf_mode"])
    
    # update differentiation parameters
    diff_config = config["differentiation"]
    diff_config["sdf_eps"] = exact(args.sdf_eps, diff_config["sdf_eps"])
    diff_config["sdf_deriv_eps"] = exact(args.sdf_deriv_eps, diff_config["sdf_deriv_eps"])
    
    # set global parameters
    utils.ITX_EPS = render_config["itx_eps"]
    utils.RAY_EPS = render_config["ray_eps"]
    utils.HIDE_ENV = Mask(render_config["hide_env"])
    if render_config["background"] == "black":
        utils.BACKGROUND = black
    else:
        utils.BACKGROUND = white
    utils.SDF_MODE = render_config["sdf_mode"]
    utils.SDF_EPS = diff_config["sdf_eps"]
    utils.SDF_DERIV_EPS = diff_config["sdf_deriv_eps"]
    utils.NUM_REFINE = render_config["num_refine"]
    utils.ITX_REFINE = render_config["itx_refine"]
    utils.GRZ_MODE = diff_config["grz_mode"]
    utils.NUM_GRZ = diff_config["num_grz"]
    
    # ------------------------------ LOAD SCENE -------------------------------
    # create exp folder
    exp_dir = join(OUT_DIR, exp_config["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    print(f"----- start experiment: {exp_config['exp_name']} -----")
    
    # load scene
    scene : mi.Scene = mi.load_file(
        os.path.join(DATA_DIR, exp_config["scene_name"], "scene.xml"),
        HIDE_ENV=render_config["hide_env"]
    )

    # hardcode position for now
    radius = 1.3
    if exp_config["scene_name"] == 'hotdog':
        height = 0.5 
    elif exp_config["scene_name"] == 'lego':
        height = 0.35
    elif exp_config["scene_name"] == 'chair':
        height = 0.6
        radius = 1.5
    else:
        height = 0.4

    # load sensors
    sensors = get_h2_sensors(
        num_sensor_theta=exp_config["num_sensor_theta"], 
        num_sensor_phi=exp_config["num_sensor_phi"],
        resx=render_config["resx"], resy=render_config["resy"],
        radius=radius, height=height
    )
    num_sensor = len(sensors)

    # ------------------------------ REF IMAGE ------------------------------
    print("----- render ref image -----")

    # create folder
    ref_dir = join(exp_dir, "ref")
    os.makedirs(ref_dir, exist_ok=True)

    # load ref integrator
    ref_integrator = mi.load_dict({"type" : f"{render_config['ref_integrator']}"})

    # dummy sdf
    sdf = SDF()

    # render ref image
    ref_images = []
    with dr.suspend_grad():
        for i in range(num_sensor):
            # mitsuba renderer
            # ref_image = mi.render(scene=scene, sensor=sensors[i], spp=render_config["ref_spp"]) 
            # custom renderer
            ref_image = ref_integrator.render(scene=scene, sdf=sdf, sensor=sensors[i], spp=render_config["ref_spp"]) 
            ref_bitmap = mi.util.convert_to_bitmap(ref_image)
            mi.util.write_bitmap(join(ref_dir, f"image{i:02d}.png"), ref_bitmap)
            ref_images.append(ref_image)

    print("----- finish rendering reference image -----")

    # ------------------------------ OPTIMIZE ------------------------------
    print("----- start optimizing -----")

    # create opt folder
    opt_dir = join(exp_dir, "opt")
    os.makedirs(opt_dir, exist_ok=True)

    # create sdf folder
    sdf_dir = join(exp_dir, "params")
    os.makedirs(sdf_dir, exist_ok=True)

    # load opt integrator
    sdf_integrator = mi.load_dict({"type" : f"{render_config['sdf_integrator']}"})

    # load init sdf
    sdf : SDF = SphereSDF(
        res=exp_config["sdf_res"], 
        radius=exp_config["sphere_radius"]
    )
    dr.enable_grad(sdf.grid.tensor())

    params = mi.traverse(scene)
    exp_config["opt_albedo"] = 'main-bsdf.base_color.volume.data' in params or 'main-bsdf.reflectance.volume.data' in params
    exp_config["opt_roughness"] = 'main-bsdf.roughness.volume.data' in params
    exp_config["opt_metallic"] = 'main-bsdf.metallic.volume.data' in params

    if exp_config["opt_albedo"]:
        if exp_config["opt_roughness"]:
            # upsample albedo resolution to sdf_res
            albedo_name = 'main-bsdf.base_color.volume.data'
            params = mi.traverse(scene)
            params[albedo_name] = upsample_to(params[albedo_name], exp_config["sdf_res"])
            params.update()
            
            # upsample roughness resolution to sdf_res
            roughness_name = 'main-bsdf.roughness.volume.data'
            params = mi.traverse(scene)
            params[roughness_name] = upsample_to(params[roughness_name], exp_config["sdf_res"])
            params.update()
            
            if exp_config["opt_metallic"]:
                # upsample metallic resolution to sdf_res
                metallic_name = 'main-bsdf.metallic.volume.data'
                params = mi.traverse(scene)
                params[metallic_name] = upsample_to(params[metallic_name], exp_config["sdf_res"])
                params.update()
        else:
            # upsample albedo resolution to sdf_res
            albedo_name = 'main-bsdf.reflectance.volume.data'
            params = mi.traverse(scene)
            params[albedo_name] = upsample_to(params[albedo_name], exp_config["sdf_res"])
            params.update()

    # load optimizer
    lr = exp_config["lr"]
    optimizer  = mi.ad.Adam(lr=lr)
    
    sdf_params = mi.traverse(sdf) 
    optimizer['grid'] = sdf_params['grid']
    
    scene_params = mi.traverse(scene)
    if exp_config["opt_albedo"]:
        optimizer[albedo_name] = scene_params[albedo_name] 
        optimizer.set_learning_rate({albedo_name: lr * exp_config["lr_albedo_scale"]}) 
        
    if exp_config["opt_roughness"]:
        optimizer[roughness_name] = scene_params[roughness_name] 
        optimizer.set_learning_rate({roughness_name: lr * exp_config["lr_roughness_scale"]}) 
        
    if exp_config["opt_metallic"]:
        optimizer[metallic_name] = scene_params[metallic_name] 
        optimizer.set_learning_rate({metallic_name: lr * exp_config["lr_metallic_scale"]})

    # set batch size
    steps = int(np.ceil(num_sensor / exp_config["batch_size"]))

    # record loss and time
    losses = []
    times = []
    times.append(time.time())

    pbar = tqdm(range(exp_config["num_epoch"]))
    for epoch in pbar:
        # sum of losses in this epoch
        loss_sum = Float(0)
        
        # batch indices
        num_indices = min(exp_config["batch_size"], num_sensor)
        indices = [(j * steps + epoch % steps) % num_sensor for j in range(num_indices)]
        np.random.shuffle(indices) # shuffle indices
        
        for i in indices:
            # render image
            seed = epoch * num_sensor + i
            image = sdf_integrator.render(
                scene=scene, sdf=sdf, sensor=sensors[i], 
                spp=render_config["spp"], spp_grad=render_config["spp_grad"], 
                seed=seed, params=scene_params
            )
            bitmap = mi.util.convert_to_bitmap(image)
            
            # save image
            if epoch % 10 == 0:
                mi.util.write_bitmap(join(opt_dir, f"image-{epoch}-{i}.png"), bitmap)
            
            # compute image loss
            loss = l1_loss(image, ref_images[i]) / exp_config["batch_size"]
            dr.backward(loss)
            loss_sum += loss
        
        # compute reg loss
        reg_loss = exp_config["laplacian_weight"] * laplacian_loss(sdf.grid.tensor())
        dr.backward(reg_loss)
        loss_sum += reg_loss
    
        # update parameters
        optimizer.step() 
        
        # check redistance
        grid = redistance(optimizer['grid'], method='fastsweep')
        grid = atleast_4d(TensorXf(grid))
        optimizer['grid'] = grid 
        
        # update sdf
        sdf_params.update(optimizer)
        scene_params.update(optimizer)
        
        # check upsample
        if epoch in exp_config["upsample"]:            
            # update learning rate
            lr /= exp_config["lr_decay"]
            optimizer.set_learning_rate({'grid': lr})
            if exp_config["opt_albedo"]:
                optimizer.set_learning_rate({albedo_name: lr * exp_config["lr_albedo_scale"]})
            if exp_config["opt_roughness"]:
                optimizer.set_learning_rate({roughness_name: lr * exp_config["lr_roughness_scale"]})
            if exp_config["opt_metallic"]:
                optimizer.set_learning_rate({metallic_name: lr * exp_config["lr_metallic_scale"]})
                
            # upsample sdf
            optimizer['grid'] = upsample_sdf(sdf, optimizer['grid'])
            
            # upsample albedo
            if exp_config["opt_albedo"] and resolution != 256:
                optimizer[albedo_name] = upsample_grid(optimizer[albedo_name])
            
            # upsample roughness
            if exp_config["opt_roughness"] and resolution != 256:
                optimizer[roughness_name] = upsample_grid(optimizer[roughness_name])
            
            # upsample metallic
            if exp_config["opt_metallic"] and resolution != 256:
                optimizer[metallic_name] = upsample_grid(optimizer[metallic_name])
        
        # decrease lr near the end
        if epoch in exp_config["lr_downsample"]:
            lr /= exp_config["lr_decay"]
            optimizer.set_learning_rate({'grid': lr})
            if exp_config["opt_albedo"]:
                optimizer.set_learning_rate({albedo_name: lr * exp_config["lr_albedo_scale"]})
            if exp_config["opt_roughness"]:
                optimizer.set_learning_rate({roughness_name: lr * exp_config["lr_roughness_scale"]})
                
        if epoch % exp_config["checkpoint"] == 0:
            # save checkpoint sdf
            resolution = sdf.grid.shape[0]
            grid = sdf.grid.tensor()
            grid = np.array(grid).flatten()
            grid = [resolution] + grid.tolist()
            np.save(join(sdf_dir, f"sdf_epoch={epoch}.npy"), grid)
            
            # save checkpoint albedo
            if exp_config["opt_albedo"]:
                resolution = optimizer[albedo_name].shape[0]
                grid = optimizer[albedo_name]
                grid = np.array(grid).flatten()
                grid = [resolution] + grid.tolist()
                np.save(join(sdf_dir, f"albedo_epoch={epoch}.npy"), grid)
            
            # save checkpoint roughness
            if exp_config["opt_roughness"]:
                resolution = optimizer[roughness_name].shape[0]
                grid = optimizer[roughness_name]
                grid = np.array(grid).flatten()
                grid = [resolution] + grid.tolist()
                np.save(join(sdf_dir, f"roughness_epoch={epoch}.npy"), grid)
                
            # save checkpoint metallic
            if exp_config["opt_metallic"]:
                resolution = optimizer[metallic_name].shape[0]
                grid = optimizer[metallic_name]
                grid = np.array(grid).flatten()
                grid = [resolution] + grid.tolist()
                np.save(join(sdf_dir, f"metallic_epoch={epoch}.npy"), grid)
        
        # record time
        times.append(time.time())
        
        # record loss
        losses.append(loss_sum[0])
        
        # print loss to progress bar
        img_loss_mean = loss_sum / num_indices
        pbar.set_description(f"img loss {1e3*img_loss_mean[0]:.2f}")

    times.append(time.time())
    print("----- finish optimizing -----")

    # ----------------------- SAVE PARAMS -----------------------
    # save config
    with open(join(exp_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
        json.dump(losses, f, indent=4)
        json.dump(times, f, indent=4)
    
    # save sdf
    resolution = sdf.grid.shape[0]
    grid = sdf.grid.tensor()
    grid = np.array(grid).flatten()
    grid = [resolution] + grid.tolist()
    np.save(join(sdf_dir, f"sdf_final.npy"), grid)

    # save albedo
    if exp_config["opt_albedo"]:
        resolution = optimizer[albedo_name].shape[0]
        grid = optimizer[albedo_name]
        grid = np.array(grid).flatten()
        grid = [resolution] + grid.tolist()
        np.save(join(sdf_dir, f"albedo_final.npy"), grid)
        vol_grid = mi.VolumeGrid(optimizer[albedo_name])
        vol_grid.write(join(sdf_dir, f"albedo_final.vol"))

    # save roughness
    if exp_config["opt_roughness"]:
        resolution = optimizer[roughness_name].shape[0]
        grid = optimizer[roughness_name]
        grid = np.array(grid).flatten()
        grid = [resolution] + grid.tolist()
        np.save(join(sdf_dir, f"roughness_final.npy"), grid)
        vol_grid = mi.VolumeGrid(optimizer[roughness_name])
        vol_grid.write(join(sdf_dir, f"roughness_final.vol"))

    # save metallic
    if exp_config["opt_metallic"]:
        resolution = optimizer[metallic_name].shape[0]
        grid = optimizer[metallic_name]
        grid = np.array(grid).flatten()
        grid = [resolution] + grid.tolist()
        np.save(join(sdf_dir, f"metallic_final.npy"), grid)
        vol_grid = mi.VolumeGrid(optimizer[metallic_name])
        vol_grid.write(join(sdf_dir, f"metallic_final.vol"))

    # ----------------------- SAVE VIDEO -----------------------
    print("----- start rendering video -----")

    # create folder
    video_dir = join(exp_dir, "video")
    os.makedirs(video_dir, exist_ok=True)

    # optimization video
    opt_video(frames_dir=opt_dir, video_dir=video_dir, fps=24)

    # rotation video
    rotate_video(video_dir=video_dir, scene=scene, sdf=sdf, params=scene_params)

    # geometry video
    geometry_video(video_dir=video_dir, scene=scene, sdf=sdf)

    print("----- finish rendering video -----")

    # ----------------------- PLOT LOSS -----------------------
    print("----- start plotting loss -----")

    # plot loss
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("epoch")
    plt.ylabel("L1 loss")
    plt.title(exp_config["exp_name"])

    # save plot
    plot_dir = join(exp_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(join(plot_dir, "loss.pdf"))
    plt.savefig(join(plot_dir, "loss.png"))

    # save loss
    np.save(join(plot_dir, "loss.npy"), losses)

    print("----- finish plotting loss -----")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', 
        type=str, default='default', 
        help='Config file name without .json extension'
    )

    # Experiment and scene name
    parser.add_argument('--name', 
        type=str, default=None, 
        help='Experiment name'
    )
    parser.add_argument('--scene', 
        type=str, required=True,
        help='Scene name that matches the directory in data/'
    )
    
    # Optimization parameters
    parser.add_argument('--num_epochs', 
        type=int, default=None, 
        help='Number of epochs'
    )
    parser.add_argument('--lr', 
        type=float, default=None, 
        help='Learning rate'
    )
    parser.add_argument('--num_sensor_theta',
        type=int, default=None,
        help='Number of theta sensors'
    )
    parser.add_argument('--num_sensor_phi',
        type=int, default=None,
        help='Number of phi sensors'
    )
    parser.add_argument('--batch_size',
        type=int, default=None,
        help='Batch size'
    )

    # Rendering parameters
    parser.add_argument('--integrator',
        type=str, default=None,
        help='Integrator type'
    )
    parser.add_argument('--hide_env',
        type=bool, default=None,
        help='Hide environment lighting'
    )
    parser.add_argument('--sdf_mode',
        type=str, default=None,
        help='SDF interpolation method'
    )
    parser.add_argument('--resx', 
        type=int, default=None, 
        help='Resolution x'
    )
    parser.add_argument('--resy', 
        type=int, default=None, 
        help='Resolution y'
    )
    parser.add_argument('--spp', 
        type=int, default=None, 
        help='Samples per pixel'
    )
    parser.add_argument('--spp_grad', 
        type=int, default=None, 
        help='Samples per pixel for gradients'
    )
    
    # Differentiation parameters
    parser.add_argument('--sdf_eps', 
        type=float, default=None, 
        help='SDF epsilon for sampling relaxed silhouette points'
    )
    parser.add_argument('--sdf_deriv_eps',
        type=float, default=None,
        help='SDF derivative epsilon for sampling relaxed silhouette points'
    )

    args = parser.parse_args()
    main(args)
