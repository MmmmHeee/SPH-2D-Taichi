'''
Render part
Currently it simply renders particles
'''
import taichi as ti
import numpy as np

__SCREEN_RES = None # screen resolution tuple: (width, height)
__WORLD_SCREEN_RATIO = None # ratio to convert from world pos to screen pos
GUI = None

__BG_COLOR = 0x112244
__LIQUID_COLOR = 0x008888
__BOUNDARY_COLOR = 0xcc6666
__PARTICLE_RADIUS = 2.0 # particle radius in pixel level

def init(res, world_screen_ratio, title='SPH2D'):
    global __SCREEN_RES
    global __WORLD_SCREEN_RATIO
    global GUI
    __SCREEN_RES = res
    __WORLD_SCREEN_RATIO = world_screen_ratio
    GUI = ti.GUI(title, __SCREEN_RES)

def renderParticles(circle_positions: np.ndarray, colors: np.ndarray):
    GUI.canvas.clear(__BG_COLOR)
    GUI.circles(circle_positions, radius=__PARTICLE_RADIUS, color=colors)

def liquidBoundToColors(particle_is_liquid: ti.var) -> np.ndarray:
    # seperate liquid and boundary with different color
    is_liquid = particle_is_liquid.to_numpy()
    colors = is_liquid * (__LIQUID_COLOR - __BOUNDARY_COLOR) + __BOUNDARY_COLOR
    return colors

def taichiToRenderPos(particle_pos: ti.Vector) -> np.ndarray:
    # convert positions in simulation world to screen world
    pos_np = particle_pos.to_numpy() * __WORLD_SCREEN_RATIO
    for pos in pos_np:
        for j in range(2):
            pos[j] /= __SCREEN_RES[j]

    return pos_np
