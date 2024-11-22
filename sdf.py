import utils
from utils import *    

class SDF(mi.Object):
    """ Signed Distance Function of an object defined on 3D voxel grid.
        param res: resolution of the voxel grid
        param grid: voxel grid representing the SDF 
        param bbox: bounding box of the voxel grid"""
    
    def __init__(self):
        mi.Object.__init__(self)
        # load bounding box
        self.bbox = mi.BoundingBox3f(Point3f(0, 0, 0), Point3f(1, 1, 1))
    
    def load(self, data):
        # if data end with .sdf
        if data.endswith('.sdf'):
            data = np.loadtxt(data) # load data
            res = int(data[0]) # load res
            self.res = res
            grid = data[1:] # load grid
            grid = mi.TensorXf(grid, shape=(res,res,res,1))
            self.grid = mi.Texture3f(grid, use_accel=False)
        if data.endswith('.npy'):
            data = np.load(data) # load data
            res = int(data[0]) # load res
            self.res = res
            grid = data[1:] # load grid
            grid = mi.TensorXf(grid, shape=(res,res,res,1))
            self.grid = mi.Texture3f(grid, use_accel=False)
        elif data.endswith('.vol'):
            data = mi.FileResolver().resolve(data)
            data = (mi.TensorXf(mi.VolumeGrid(data)))
            self.grid = mi.Texture3f(data.shape[:3], 1, use_accel=False)
            if data.ndim == 3:
                data = data[..., None]
            self.grid.set_tensor(data, migrate=False)
            # self.texture.set_tensor(atleast_4d(data), migrate=False)
    
    def in_range(self, p:Point3f) -> Bool:
        """Return whether a point is in range."""
        return dr.all(p > 0) & dr.all(p < 1)

    def at(self, p:Point3f) -> Float:
        """Return the signed distance at a point."""
        method = utils.SDF_MODE
        if method == 'linear':
            return self.grid.eval(p)[0]
        elif method == 'cubic':
            return self.grid.eval_cubic(p)[0]
        else:
            raise ValueError(f"This SDF method ({method}) is not supported.")

    def gradient(self, p:Point3f) -> Vector3f:
        """Return the gradient of the signed distance at a point."""
        method = utils.SDF_MODE
        if method == 'linear':
            epsilon = 1e-5 # finite difference
            dx = Vector3f(epsilon, 0, 0)
            dy = Vector3f(0, epsilon, 0)
            dz = Vector3f(0, 0, epsilon)
            dfdx = (self.at(p + dx) - self.at(p - dx)) / (2 * epsilon)
            dfdy = (self.at(p + dy) - self.at(p - dy)) / (2 * epsilon)
            dfdz = (self.at(p + dz) - self.at(p - dz)) / (2 * epsilon)
            return Vector3f(dfdx, dfdy, dfdz)
        elif method == 'cubic':
            return self.grid.eval_cubic_grad(p)[1][0]
        else:
            raise ValueError(f"This SDF method ({method}) is not supported.")
        
    def dir_deriv(self, p:Point3f, d:Vector3f) -> Float:
        """Return the directional derivative of the signed distance at a point."""
        grad = dr.normalize(self.gradient(p))
        return dr.dot(grad, d)

    # ------------------ GET GRAZING POINT ------------------
    def get_grz_secant(self, ray:Ray3f, t_min, t_max) -> Point3f:
        """Secant method for finding the grazing point."""
        d = ray.d
        L = dr.detach(ray(t_min))
        R = dr.detach(ray(t_max))
        
        cnt = Int(0) # >6
        loop = mi.Loop(name="get grazing point using secant method", 
                    state=lambda: (L, R, cnt))
        with dr.suspend_grad():
            while loop(cnt < utils.NUM_GRZ): 
                dL = self.dir_deriv(L, d)
                dR = self.dir_deriv(R, d)
                M = R - dR * (R - L) / (dR - dL)
                dM = self.dir_deriv(M, d)
                L[dM < 0] = M
                R[dM > 0] = M
                cnt += 1
        
        return dr.detach(L)

    def get_grz_bisection(self, ray:Ray3f, t_min, t_max) -> Point3f:
        """Bisection method for finding the grazing point."""
        d = ray.d
        L = dr.detach(ray(t_min))
        R = dr.detach(ray(t_max))
        # M = dr.detach((L + R) / 2)
        
        cnt = Int(0) # >12
        loop = mi.Loop(name="get grazing point using bisection method", 
                       state=lambda: (L, R, cnt))
        with dr.suspend_grad():
            while loop(cnt < utils.NUM_GRZ): 
                M = (L + R) / 2
                dM = self.dir_deriv(M, d)
                L[dM < 0] = M[dM < 0]
                R[dM > 0] = M[dM > 0]
                cnt += 1
        
        return dr.detach(L)

    def get_grz(self, ray:Ray3f, t_min, t_max) -> Point3f:
        method = utils.GRZ_MODE
        if method == 'secant':
            return self.get_grz_secant(ray, t_min, t_max)
        elif method == 'bisection':
            return self.get_grz_bisection(ray, t_min, t_max)
        else:
            raise ValueError(f"This grz method ({method}) is not supported.")
    
    # ------------------ GET RAY INTERSECTION ------------------
    def refine_intersect(self, ray:Ray3f, t_min, t_max) -> Point3f:
        """Bisection method for refining the intersection point."""
        if utils.NUM_REFINE == 0:
            return dr.detach((t_min + t_max) / 2)
        
        L = dr.detach(t_min)
        R = dr.detach(t_max)
        M = dr.detach((L + R) / 2)
        
        cnt = Int(0) # >20
        loop = mi.Loop(name="refine intersection", 
                       state=lambda: (L, R, M, cnt))
        with dr.suspend_grad():
            while loop(cnt < utils.NUM_REFINE): 
                # update endpoints
                dL = self.at(ray(L))
                dR = self.at(ray(R))
                
                # bisection step
                M = (L + R) / 2
                dM = self.at(ray(M))
                L[dM > 0] = M[dM > 0]
                R[dM < 0] = M[dM < 0]
                cnt += 1
        
        return dr.detach(M)
    
    def get_intersect(self, ray:Ray3f, t:Float, is_valid:Mask) -> SurfaceInteraction3f:
        """ Return the surface interaction without supporting SDF autodiff.
            Code adapted from https://github.com/rgl-epfl/differentiable-sdf-rendering """
        si = dr.zeros(SurfaceInteraction3f)
        si.t = t
        si.p = ray(t)
        si.sh_frame.n = dr.normalize(self.gradient(si.p))
        si.initialize_sh_frame()
        si.n = si.sh_frame.n
        si.wi = dr.select(si.is_valid(), si.to_local(-ray.d), -ray.d)
        si.wavelengths = ray.wavelengths
        # si.uv = si.sh_frame.to_uv(si.p)
        si.dp_du = si.sh_frame.s
        si.dp_dv = si.sh_frame.t
        return si

    def ray_intersect_default(self, ray:Ray3f) -> Float:
        """ Default sphere tracing for ray intersection time. 
            Return the time of intersection. """        
        # copy ray
        ray = Ray3f(ray) 
        
        # parameters for sphere tracing
        t = Float(0)
        no_itx = Mask(True)
        in_range = Mask(True)
        
        # check bounding box
        itx_bbox, mint, maxt = self.bbox.ray_intersect(ray)
        inside_bbox = self.bbox.contains(ray.o)
        ray.maxt = dr.minimum(maxt, ray.maxt)
        t = dr.select(inside_bbox, 0, mint + RAY_EPS)
        t = dr.detach(t)
        
        # normal sphere tracing
        loop = mi.Loop(name="normal sphere tracing", 
                       state=lambda: (t, no_itx, in_range))
        with dr.suspend_grad():
            while loop(no_itx & in_range):
                # update point
                p = ray(t)
                d = self.at(p)
                
                # sphere trace one step
                t += d
                no_itx = (d > ITX_EPS)
                in_range = self.in_range(p) & (t < ray.maxt)
        
        # check valid ray
        is_valid = (~no_itx) & in_range
        t[~is_valid] = dr.inf
        t[is_valid] = self.refine_intersect(ray, t - ITX_REFINE, t + ITX_REFINE)
        
        return t, is_valid

    def ray_intersect_grz(self, ray:Ray3f):
        """ Sphere tracing for ray intersection and grazing point.
            Return the time of intersection. """
        # copy ray
        ray = Ray3f(ray) 
        
        # parameters for sphere tracing
        t = Float(0)
        no_itx = Mask(True)
        in_range = Mask(True)
        
        # parameters for solving grazing point
        grz_t = Float(0)
        min_sdf = Float(1)
        pos_dir = Mask(True)
    
        # check bounding box
        itx_bbox, mint, maxt = self.bbox.ray_intersect(ray)
        inside_bbox = self.bbox.contains(ray.o)
        ray.maxt = dr.minimum(maxt, ray.maxt)
        t = dr.select(inside_bbox, 0, mint + RAY_EPS)
        t = dr.detach(t)

        loop = mi.Loop(
            name="sphere tracing for grazing point", 
            state=lambda: (t, no_itx, in_range, pos_dir, grz_t, min_sdf)
        )
        with dr.suspend_grad():
            while loop(no_itx & in_range):
            # while dr.any(no_itx & in_range):
                # update point
                p = Point3f(ray(t))
                d = self.at(p)
                dir = self.dir_deriv(p, ray.d)
                
                # check boundary path
                idx = (~pos_dir) & (dir > 0) & (d < min_sdf) & (d > ITX_EPS)
                grz_t[idx] = t
                min_sdf[idx] = d
                
                # sphere trace one step
                t += d
                pos_dir = (dir > 0)
                no_itx = (d > ITX_EPS)
                in_range = self.in_range(p) & (t < ray.maxt)
            
            # fine-tune grazing point
            grz = self.get_grz(ray, grz_t - min_sdf, grz_t + min_sdf) 
            # grz = ray(grz_t) # no fine-tune and do not work
            
            # check grazing point
            is_grz = (dr.abs(self.at(grz)) < SDF_EPS) & \
                     (dr.abs(self.dir_deriv(grz, ray.d)) < SDF_DERIV_EPS) & \
                     (self.at(grz) > ITX_EPS)
        
        # check valid ray
        is_valid = (~no_itx) & in_range
        t[~is_valid] = dr.inf
        t[is_valid] = self.refine_intersect(ray, t - ITX_REFINE, t + ITX_REFINE)
        
        return t, is_valid, is_grz, grz
    
    def ray_intersect(self, ray:Ray3f, method='default') -> mi.SurfaceInteraction3f:
        """ Return the intersection with a ray. """
        if method == 'default': # interior
            t, is_valid = self.ray_intersect_default(ray)
            return self.get_intersect(ray, t, is_valid)
        elif method == 'grz': # interior + boundary
            t, is_valid, is_bp, bp = self.ray_intersect_grz(ray)
            return self.get_intersect(ray, t, is_valid), is_bp, bp
        else:
            raise ValueError(f"This ray intersect method ({method}) is not supported.")
    
    def traverse(self, callback):
        callback.put_parameter("grid", self.grid.tensor(), mi.ParamFlags.Differentiable)
        
    
    def parameters_changed(self, keys):
        self.grid.set_tensor(self.grid.tensor())


class TransformedSDF(SDF):
    """ Transformed Signed Distance Function. 
        param sdf: SDF to be transformed
        param orgin: origin of the local and world space
        param to_world: transformation from local to world space
        param to_local: transformation from world to local space """
    
    def __init__(self, sdf:SDF, transform:mi.Transform4f):
        super().__init__()
        self.sdf = sdf
        self.orgin = Point3f(0.5, 0.5, 0.5)
        self.to_world = transform
        self.to_local = transform.inverse()
    
    def load(self, data):
        self.sdf.load(data)
    
    def at(self, p:Point3f) -> Float:
        """Return the signed distance at a point."""
        p_local = self.to_local @ (p - self.orgin) + self.orgin
        return self.sdf.at(p_local)
    
    def gradient(self, p:Point3f) -> Vector3f:
        """Return the gradient of the signed distance at a point."""
        p_local = self.to_local @ (p - self.orgin) + self.orgin
        grad_local = self.sdf.gradient(p_local)
        return self.to_world @ mi.Normal3f(grad_local)

    def traverse(self, callback):
        callback.put_parameter("grid", self.sdf.grid.tensor(), mi.ParamFlags.Differentiable)
    
    def parameters_changed(self, keys):
        self.grid.set_tensor(self.grid.tensor())

class SphereSDF(SDF):
    """ Sphere Signed Distance Function. 
        Used for initial SDF for optimization. """
    
    def __init__(self, res=256, center:Point3f=Point3f(0.5,0.5,0.5), radius:Float=0.2):
        super().__init__()
        
        def dist_to_sphere(p:Point3f, center:Point3f, radius:Float) -> Float:
            """Signed distance function for a sphere."""
            # return - dr.norm(p - center) + radius
            return dr.norm(p - center) - radius
        
        self.res = res
        grid = dist_to_sphere(dr.meshgrid(
            dr.linspace(Float, 0, 1, res),
            dr.linspace(Float, 0, 1, res),
            dr.linspace(Float, 0, 1, res)
        ), center=center, radius=radius)
        grid = mi.TensorXf(grid, shape=(res,res,res,1))
        self.grid = mi.Texture3f(grid, use_accel=False)
   
class PlaneSDF(SDF):
    """ Signed Distance Function of an object and a plane. """

    def __init__(self, sdf:SDF, origin:Point3f, target:Point3f):
        super().__init__()
        self.sdf = sdf
        self.origin = origin
        self.normal = dr.normalize(target - origin)

    def load(self, data):
        self.sdf.load(data)

    def at(self, p:Point3f) -> Float:
        """Return the signed distance at a point."""
        dist_object = self.sdf.at(p)
        dist_plane = dr.dot(p - self.origin, self.normal)
        return dr.select(dist_object < dist_plane, dist_object, dist_plane)
    
    def gradient(self, p:Point3f) -> Vector3f:
        """Return the gradient of the signed distance at a point."""
        dist_object = self.sdf.at(p)
        dist_plane = dr.dot(p - self.origin, self.normal)
        return dr.select(
            dist_object < dist_plane, 
            self.sdf.gradient(p),
            self.normal
        )
    
    def traverse(self, callback):
        callback.put_parameter("grid", self.sdf.grid.tensor(), mi.ParamFlags.Differentiable)
    
    def parameters_changed(self, keys):
        self.sdf.grid.set_tensor(self.sdf.grid.tensor())


class CombinedSDF(SDF):
    """ Combined Signed Distance Function. """

    def __init__(self, sdf1:SDF, sdf2:SDF, op='union'):
        super().__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.op = op
        
    def at(self, p:Point3f) -> Float:
        """Return the signed distance at a point."""
        dist1 = self.sdf1.at(p)
        dist2 = self.sdf2.at(p)
        if self.op == 'union':
            return dr.minimum(dist1, dist2)
        elif self.op == 'intersection':
            return dr.maximum(dist1, dist2)
        elif self.op == 'difference':
            return dr.maximum(dist1, -dist2)
        else:
            raise ValueError(f"This operation ({self.op}) is not supported.")
    
    def gradient(self, p:Point3f) -> Vector3f:
        """Return the gradient of the signed distance at a point."""
        dist1 = self.sdf1.at(p)
        dist2 = self.sdf2.at(p)
        if self.op == 'union':
            return dr.select(dist1 < dist2, self.sdf1.gradient(p), self.sdf2.gradient(p))
        elif self.op == 'intersection':
            return dr.select(dist1 > dist2, self.sdf1.gradient(p), self.sdf2.gradient(p))
        elif self.op == 'difference':
            return dr.select(dist1 > -dist2, self.sdf1.gradient(p), -self.sdf2.gradient(p))
        else:
            raise ValueError(f"This operation ({self.op}) is not supported.")
    
    def traverse(self, callback):
        callback.put_parameter("grid", self.sdf1.grid.tensor(), mi.ParamFlags.Differentiable)
        callback.put_parameter("grid", self.sdf2.grid.tensor(), mi.ParamFlags.Differentiable)
    
    def parameters_changed(self, keys):
        self.sdf1.grid.set_tensor(self.sdf1.grid.tensor())
        self.sdf2.grid.set_tensor(self.sdf2.grid.tensor())