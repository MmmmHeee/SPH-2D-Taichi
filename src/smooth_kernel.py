'''
Refer to
[Muller2003] Müller, Matthias, David Charypar, and Markus Gross. 
"Particle-based fluid simulation for interactive applications." 
Proceedings of the 2003 ACM
'''

import taichi as ti
import numpy as np
import math

__POLY6_KERNEL_FACTOR = 315 / (64 * math.pi)
__POLY6_GRAD_FACTOR = - 945 / (32 * math.pi)
__POLY6_LAP_FACTOR = 945 / (8 * math.pi)
__SPIKY_KERNEL_FACTOR = 15 / math.pi
__SPIKY_GRAD_FACTOR = - 45 / math.pi
__VIS_KERNEL_FACTOR = 15 / (2 * math.pi)
__VIS_LAP_FACTOR = 45 / math.pi
__CUBIC_KERNEL_FACTOR = 10 / (7 * math.pi)

@ti.func
def poly6Kernel(r, h):
    # \frac{315}{64 \pi h^9} (h^2 - |r|^2)^3

    res = 0.0
    r_len = r.norm()
    if r_len <= h:
        tmp = (h * h - r_len * r_len) / (h * h * h)
        res = __POLY6_KERNEL_FACTOR * tmp * tmp * tmp
    return res

@ti.func
def poly6Grad(r, h):
    # -\frac{945}{32 \pi h^9} (h^2 - |r|^2)^2 * r

    res = ti.Vector([0.0, 0.0], dt = float)
    r_len = r.norm()
    if r_len <= h:
        h2 = h * h
        tmp = (h2 - r_len * r_len) / (h2 * h2)
        res = __POLY6_GRAD_FACTOR * tmp * tmp / h * r
    return res


@ti.func
def poly6Lap(r, h):    
    # \frac{945}{8 \pi h^9} (h^2 - |r|^2)^2 (|r|^2 - \frac{3}{4}(h^2 - |r|^2))

    res = 0.0
    r_len = r.norm()
    if r_len <= h:
        h2 = h * h
        r2 = r_len * r_len
        tmp = (h2 - r2)
        tmp_div_h2 = tmp / (h2 * h2)
        tmp2 = r2 - 0.75 * tmp
        res = __POLY6_LAP_FACTOR * tmp_div_h2 * tmp_div_h2 * tmp2 / h
    return res

@ti.func
def spikyKernel(r, h):
    # \frac{15}{\pi h^6} (h - r)^3 (r^2 - \frac{3}{4} (h^2 - r^2))

    res = 0.0
    r_len = r.norm()
    if r_len <= h:
        h2 = h * h
        r2 = r_len * r_len
        tmp = (h - r) / h2
        tmp2 = (r2 - 0.75 * (h2 - r2))
        res = __SPIKY_KERNEL_FACTOR * tmp * tmp * tmp * tmp2
    return res

@ti.func
def spikyGrad(r, h):
    # -\frac{45}{\pi h^6 |r|}(h - |r|)^2 * r

    res = ti.Vector([0.0, 0.0], dt = float)
    r_len = r.norm()
    if r_len <= h:
        tmp = (h - r_len) / (h * h * h)
        res = __SPIKY_GRAD_FACTOR * tmp * tmp / r_len * r
    return res

@ti.func
def viscosityKernel(r, h):
    # \frac{15}{2 \pi h^3} (-\frac{|r|^3}{2h^3} + \frac{|r|^2}{h^2} + \frac{h}{2|r|} - 1)

    res = 0.0
    r_len = r.norm()
    if r_len <= h:
        h2 = h * h
        h3 = h2 * h
        r2 = r_len * r_len
        r3 = r2 * r_len
        
        tmp = - r3 / (2 * h3) + r2 / h2 + h / (2 * r_len) - 1
        res = __VIS_KERNEL_FACTOR * tmp / h3

    return res

@ti.func
def viscosityLap(r, h):
    # \frac{45}{\pi h^6} (h - |r|)

    res = 0.0
    r_len = r.norm()
    if r_len <= h:
        h3 = h * h * h
        res = __VIS_LAP_FACTOR / (h3 * h3) * (h - r_len) 

    return res

@ti.func
def cubicKernel(r, h):
    # value of cubic spline smoothing kernel
    r_len = r.norm()
    
    half_h = h / 2
    k = __CUBIC_KERNEL_FACTOR / half_h**2
    q = r_len / half_h

    # assert q >= 0.0

    res = 0.0
    if q <= 1.0:
        q2 = q**2
        res = k * (1 - 1.5 * q2 + 0.75 * q * q2)
    elif q < 2.0:
        res = k * 0.25 * (2 - q)**3
    return res

@ti.func
def cubicGrad(r, h):
    # derivative of cubcic spline smoothing kernel
    r_len = r.norm()
    r_dir = r.normalized()


    half_h = h / 2
    k = __CUBIC_KERNEL_FACTOR / half_h**2
    q = r_len / half_h
    
    # assert q > 0.0
    res = ti.Vector([0.0, 0.0], dt=float)
    if q < 1.0:
        res = (k / half_h) * (-3. * q + 2.25 * q**2) * r_dir
    elif q < 2.0:
        res = -0.75 * (k / half_h) * (2. - q)**2 * r_dir
    return res