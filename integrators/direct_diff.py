from utils import *
from sdf import *
from integrators.base import *

class SDFDirectDiffIntegrator(SDFBaseIntegrator):
    
    def __init__(self, prop=mi.Properties()):
        super().__init__(prop)

    def sample(self, 
        scene:mi.Scene, sdf:SDF, 
        sampler:mi.Sampler, ray:mi.Ray3f, max_depth = 2, 
        active = True, **kwargs
    ):       
        # ray intersection
        itx : SurfaceInteraction3f 
        itx, is_grz_i, grz_i = sdf.ray_intersect(ray, method='grz')
        valid_primary = itx.is_valid()

        # surface emission
        E = itx.emitter(scene=scene).eval(itx)

        # ----------------------- INTERIOR INTEGRAL ------------------------
        # n = itx.n
        n = dr.detach(itx.n)
        v = - sdf.at(itx.p) / dr.squared_norm(n) * n
        v = dr.select(valid_primary, v, Vector3f(0))
        itx.p = dr.replace_grad(itx.p, v) 
        
        # ------------------------- EVAL PATH ----------------------------
        # sample emitter
        samples = sampler.next_2d()
        itxL : mi.DirectionSample3f
        itxL, Li = scene.sample_emitter_direction(itx, samples, 
                                                  test_visibility=False)    
        Li = dr.select(valid_primary, Li, black)

        # eval bsdf
        bsdf_ctx = mi.BSDFContext()
        bsdf = scene.shapes()[0].bsdf()
        bsdf = bsdf.eval(bsdf_ctx, itx, itx.to_local(itxL.d))
        Li *= bsdf

        # check shadow ray
        rayL : Ray3f = Ray3f(itx.p, itxL.d)
        rayL.o += dr.normalize(rayL.d) * RAY_EPS
        itx_shadow : mi.SurfaceInteraction3f
        itx_shadow, is_grz_o, grz_o = sdf.ray_intersect(rayL, method='grz')
        valid_secondary = (~itx_shadow.is_valid())
        Li = dr.select(valid_secondary, Li, black)
        
        # check hide environment
        valid_ray = valid_primary #& valid_secondary
        valid_ray |= (~utils.HIDE_ENV)
        L = dr.select(valid_ray, E + Li, utils.BACKGROUND)
 
        # --------------------- EVAL SILHOUETTE ------------------------
        # ray intersection at the grazing point
        t_grz = dr.select(is_grz_i, dr.norm(grz_i - ray.o), dr.inf)       
        itx_grz = sdf.get_intersect(ray, t_grz, is_grz_i)
        
        # sample emitter
        samples = sampler.next_2d()
        itxL_grz : mi.DirectionSample3f
        itxL_grz, L_grz = scene.sample_emitter_direction(itx_grz, samples, 
                                                         test_visibility=False)
        L_grz = dr.select(itx_grz.is_valid(), L_grz, black)
        
        # eval bsdf
        bsdf_ctx = mi.BSDFContext()
        bsdf = scene.shapes()[0].bsdf()
        bsdf_grz = bsdf.eval(bsdf_ctx, itx_grz, itx_grz.to_local(itxL_grz.d))
        L_grz *= bsdf_grz
        
        # check shadow ray
        rayL_grz : Ray3f = Ray3f(itx_grz.p, itxL_grz.d)
        rayL_grz.o += dr.normalize(rayL_grz.d) * RAY_EPS
        itx_shadow_grz : mi.SurfaceInteraction3f = sdf.ray_intersect(rayL_grz)
        L_grz = dr.select(~itx_shadow_grz.is_valid(), L_grz, black)

        # -------------------- EVAL BOUNDARY INTEGRAL -----------------------
        # grazing point
        grz = dr.zeros(Point3f)
        grz[is_grz_i] = grz_i
        grz[is_grz_o] = grz_o
        
        # normal velocity at the grazing point
        vn = - sdf.at(grz)
        
        # evaluate boundary integral
        bp = dr.zeros(mi.Color3f)
        bp[is_grz_i] = vn * dr.detach(L_grz - L) / SDF_EPS
        bp[is_grz_o] = vn * dr.detach(-L) / SDF_EPS

        # L = dr.replace_grad(L, L)
        L = dr.replace_grad(L, L + bp)
        return L, valid_ray, itx.p

mi.register_integrator("sdf-direct-diff", 
                       lambda props: SDFDirectDiffIntegrator(props))