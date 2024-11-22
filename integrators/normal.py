from utils import *
from sdf import *
from integrators.base import *

class NormalIntegrator(SDFBaseIntegrator):
    
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
        
        # adjust normal direction
        itx.n = dr.select(dr.dot(itx.n, ray.d) < 0, itx.n, -itx.n)
        
        # map normal in range [-1, 1] to color in range [0, 1]
        itx.n = Vector3f(-itx.n.x, itx.n.y, -itx.n.z)
        L += dr.select(valid_primary, (itx.n + 1) / 2, black)

        # check hide environment
        valid_primary |= (~utils.HIDE_ENV)
        
        return dr.select(valid_primary, L, utils.BACKGROUND), valid_primary, itx.p

mi.register_integrator("normal", lambda props: NormalIntegrator(props))


class SDFNormalIntegrator(SDFBaseIntegrator):
    
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
        
        # map normal in range [-1, 1] to color in range [0, 1]
        itx.n = Vector3f(-itx.n.x, itx.n.y, -itx.n.z)
        L += dr.select(valid_primary, (itx.n + 1) / 2, black)

        # check hide environment
        valid_primary |= (~utils.HIDE_ENV)
        
        return dr.select(valid_primary, L, utils.BACKGROUND), valid_primary, itx.p

mi.register_integrator("sdf-normal", lambda props: SDFNormalIntegrator(props))