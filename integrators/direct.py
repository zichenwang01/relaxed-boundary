from utils import *
from sdf import *
from integrators.base import *

class DirectIntegrator(SDFBaseIntegrator):
    
    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)

    def sample(self, 
        scene:mi.Scene, sdf:SDF, 
        sampler:mi.Sampler, ray:mi.Ray3f, max_depth = 2, 
        active = True, **kwargs
    ):       
        # ray intersection
        itx : SurfaceInteraction3f = scene.ray_intersect(ray)
        valid_primary = itx.is_valid()

        # surface emission
        L = itx.emitter(scene=scene).eval(itx)

        # sample emitter
        itxL : mi.DirectionSample3f
        Li : mi.Color3f
        itxL, Li = scene.sample_emitter_direction(itx, sampler.next_2d(), 
                                                 test_visibility=False)   
        
        # eval bsdf
        bsdf_ctx = mi.BSDFContext()
        bsdf = itx.bsdf()
        bsdf = bsdf.eval(bsdf_ctx, itx, itx.to_local(itxL.d))
        Li *= bsdf

        # check shadow ray
        rayL : Ray3f = Ray3f(itx.p, itxL.d)
        rayL.o += dr.normalize(rayL.d) * RAY_EPS
        itx_shadow : mi.SurfaceInteraction3f = scene.ray_intersect(rayL)
        valid_secondary = (~itx_shadow.is_valid()) \
                        | (dr.norm(itx_shadow.p - itxL.p) < ITX_EPS)
        
        # direct illumination
        valid_ray = valid_primary & valid_secondary
        L += dr.select(valid_ray, Li, black)
        
        # check hide environment
        valid_primary |= (~utils.HIDE_ENV)
        
        return dr.select(valid_primary, L, utils.BACKGROUND), valid_primary, itx.p

mi.register_integrator("direct", lambda props: DirectIntegrator(props))


class SDFDirectIntegrator(SDFBaseIntegrator):
    
    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)

    def sample(self, 
        scene:mi.Scene, sdf:SDF, 
        sampler:mi.Sampler, ray:mi.Ray3f, max_depth = 2, 
        active = True, **kwargs
    ):               
        # ray intersection
        itx : SurfaceInteraction3f = sdf.ray_intersect(ray)
        valid_primary = itx.is_valid()

        # surface emission
        L = itx.emitter(scene=scene).eval(itx)

        # sample emitter
        itxL : mi.DirectionSample3f
        Li : mi.Color3f
        itxL, Li = scene.sample_emitter_direction(itx, sampler.next_2d(), 
                                                 test_visibility=False)   
        
        # eval bsdf
        bsdf_ctx = mi.BSDFContext()
        bsdf = scene.shapes()[0].bsdf()
        bsdf = bsdf.eval(bsdf_ctx, itx, itx.to_local(itxL.d))
        Li *= bsdf

        # check shadow ray
        rayL : Ray3f = Ray3f(itx.p, itxL.d)
        rayL.o += dr.normalize(rayL.d) * RAY_EPS
        itx_shadow : mi.SurfaceInteraction3f = sdf.ray_intersect(rayL)
        valid_secondary = (~itx_shadow.is_valid()) \
                        | (dr.norm(itx_shadow.p - itxL.p) < ITX_EPS)
        
        # direct illumination
        L += dr.select(valid_secondary, Li, black)
        # L += Li
        
        # check hide environment
        valid_primary |= (~utils.HIDE_ENV)
        
        return dr.select(valid_primary, L, utils.BACKGROUND), valid_primary, itx.p

mi.register_integrator("sdf-direct", lambda props: SDFDirectIntegrator(props))