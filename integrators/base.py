import gc
from utils import *
from sdf import SDF

class SDFBaseIntegrator(mi.SamplingIntegrator):
    
    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)
    
    def sample(self, 
        scene:mi.Scene, sdf:SDF, 
        sampler:mi.Sampler, ray:mi.Ray3f, max_depth = 2, 
        active = True, **kwargs
    ):        
        pass
        
    def prepare_sampler(self, sensor, film_size, seed, spp):
        """Prepare a sampler forked from the sensor sampler."""
        """Code developed from https://github.com/rgl-epfl/differentiable-sdf-rendering"""
        wavefront_size = dr.prod(film_size) * spp
        sampler = sensor.sampler().clone()
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp)
        sampler.seed(seed, wavefront_size)
        return sampler
    
    def sample_ray(self, sensor, sampler, spp):
        """ Sample a batch of rays from the sensor.
            Returns ray, ray_weight, position_sample """
        """ Code developed from https://github.com/rgl-epfl/differentiable-sdf-rendering """
        film = sensor.film()
        film_size = film.crop_size()
        border_size = film.rfilter().border_size()  
        if film.sample_border():
            film_size += 2 * border_size
        film_offset = film.crop_offset()
        
        # Compute the pixel index
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)
        idx //= dr.opaque(mi.UInt32, spp)
        
        # Compute the position on the image plane
        position = mi.Vector2i()
        position.y = idx // film_size[0]
        position.x = dr.fma(-film_size[0], position.y, idx)
        position += mi.Vector2i(film.crop_offset())
        if film.sample_border():
            position -= border_size
        
        # Compute the random offset within the pixel
        offset = sampler.next_2d()
        
        # Compute the sample position
        position_sample = position + offset
        
        # Rescale the sample position in [0, 1]^2
        position_rescaled = (position_sample - film_offset) / mi.Vector2f(film_size)
        
        # Shutter open time
        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()
        
        # Wavelength sample of continuous spectrum
        wavelength_sample = sampler.next_1d()
        
        # Aperture sample for depth of field
        aperture_sample = mi.Point2f(0.5)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()
        
        # Sample the ray
        ray, ray_weight = sensor.sample_ray_differential(time, wavelength_sample, position_rescaled, aperture_sample)
        
        # normalize ray direction
        ray.d = dr.normalize(ray.d)
        
        # return ray, 0, position_sample, aperture_sample
        return ray, ray_weight, position_sample, aperture_sample, wavelength_sample
 
    def render(self, 
        scene:mi.Scene, sdf:SDF, 
        sensor:mi.Sensor=None, seed=0, spp=32, max_depth=2,
        develop=True, active=True, **kwargs
    ):
        # prepare sensor
        if sensor is None:
            sensor = scene.sensors()[0]

        # prepare film
        film : mi.Film = sensor.film()
        film_size = film.crop_size()
        border_size = film.rfilter().border_size()  
        if film.sample_border():
            film_size += 2 * border_size
        film.prepare([])
        
        # prepare sampler
        sampler = self.prepare_sampler(sensor, film_size, seed, spp)

        # sample a batch of rays
        ray, ray_weight, position_sample, aperture_sample, wavelength_samlpe = self.sample_ray(sensor, sampler, spp)

        # scale the ray differentials
        diff_scale_factor = dr.rsqrt(mi.ScalarFloat(spp))
        ray.scale_differential(diff_scale_factor)
        
        # compute ray contribution
        # motion is the point that we reparametriz
        L, mask, motion = self.sample(scene, sdf, sampler, ray, max_depth)
        L *= ray_weight

        # compute image plane gradients
        local = sensor.world_transform().inverse() @ motion
        local /= local.z   
        local *= (0.5 / dr.tan(dr.pi/8)) # hardcoded fov=45
        position_grad = Vector2f(local.x, local.y)
        position_grad = film_size * (-position_grad)
        position_sample = dr.replace_grad(position_sample, position_grad)
        
        # prepare channels
        RGB = [None] * 3
        RGB[0] = L.x
        RGB[1] = L.y
        RGB[2] = L.z
        RGB.append(mi.Float(1)) # weight channel
        
        # put RGB values in the film
        block : mi.ImageBlock = sensor.film().create_block()
        block.set_normalize(False)
        block.put(position_sample, RGB, active)
        sensor.film().put_block(block)
        
        # develop the film to get the image   
        image = sensor.film().develop()  
        return image
    
    def traverse(self, callback):
        super().traverse(callback)
    
    def parameters_changed(self, keys):
        super().parameters_changed(keys)

mi.register_integrator("sdf-base", lambda props: SDFBaseIntegrator(props))