import utils
from utils import *
from sdf import *
from integrators.base import *

class AlbedoIntegrator(SDFBaseIntegrator):
    
    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)

    def sample(self, 
        scene:mi.Scene, sdf:SDF, sampler:mi.Sampler, 
        ray:mi.Ray3f, max_depth = 2, 
        active = True, **kwargs
    ):       
        # ray intersection
        itx : SurfaceInteraction3f = scene.ray_intersect(ray)
        valid_primary = itx.is_valid()
        
        # surface emission
        L = itx.emitter(scene=scene).eval(itx) 
        
        # eval albedo
        bsdf_ctx = mi.BSDFContext()
        bsdf : mi.BSDF = itx.bsdf()
        albedo = bsdf.eval_diffuse_reflectance(itx)
        L += dr.select(valid_primary, albedo, black)
        
        # check hide environment
        valid_primary |= (~utils.HIDE_ENV)
        
        return dr.select(valid_primary, L, utils.background), valid_primary, itx.p

mi.register_integrator("albedo", lambda props: AlbedoIntegrator(props))


class SDFAlbedoIntegrator(SDFBaseIntegrator):
    
    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)

    def sample(self, 
        scene:mi.Scene, sdf:SDF, sampler:mi.Sampler, 
        ray:mi.Ray3f, max_depth = 2, 
        active = True, **kwargs
    ):       
        # ray intersection
        itx : SurfaceInteraction3f = sdf.ray_intersect(ray)
        valid_primary = itx.is_valid()
        
        # surface emission
        L = itx.emitter(scene=scene).eval(itx) 
        
        # eval albedo
        bsdf_ctx = mi.BSDFContext()
        bsdf : mi.BSDF = scene.shapes()[0].bsdf()
        albedo = bsdf.eval_diffuse_reflectance(itx)
        L += dr.select(valid_primary, albedo, black)
        
        # check hide environment
        valid_primary |= (~utils.HIDE_ENV)
        
        return dr.select(valid_primary, L, utils.background), valid_primary, itx.p

mi.register_integrator("sdf-albedo", lambda props: SDFAlbedoIntegrator(props))