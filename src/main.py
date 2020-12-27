'''
Simple 2D SPH fluid simulation implemented using taichi
WCSPH and PCISPH are implemented in the project.

Change line 66~68 to switch model
'''
import taichi as ti
import numpy as np

from model import BaseSPH
from model import WCSPH
from model import PCISPH

from grid import Grid
from simulator import Simulator
from world import World, WorldGen

import renderer as R

# init taichi
# ti.init(arch=ti.gpu, default_fp = ti.f64)
ti.init(arch=ti.gpu, default_fp = ti.f64)

__SCREEN_RES = (400, 400)
__WORLD_TO_SCREEN_RATIO = 40

__DX = 0.1 # default particle distance at reference density (1000)
__H_FAC = 3 # kernel radius

def initWorld(gen: WorldGen):
    # initialize particles
    gen.add_cube_random((1, 0.8), (4, 6.8), is_fluid=1, density=1000)

    gen.add_cube((0, 0), (10, 0.5), 0, 1000)
    gen.add_cube((0, 0.5), (0.5, 10), 0, 1000)
    gen.add_cube((9.5, 0.5), (10, 10), 0, 1000)

def main():
    # calculate boundary
    bound = np.array([0, __SCREEN_RES[0], 0, __SCREEN_RES[1]]) / __WORLD_TO_SCREEN_RATIO
    bound = tuple(bound)

    boundary_width = 0.5 + __DX
    world_bound = (bound[0] + boundary_width, 
                   bound[1] - boundary_width,
                   bound[2] + boundary_width, 
                   bound[3] - boundary_width)

    # init renderer
    R.init(__SCREEN_RES, __WORLD_TO_SCREEN_RATIO)

    # init world
    gen = WorldGen(__DX)
    initWorld(gen)
    particle_num = gen.getParticleNum()
    world = World(particle_num)

    # init grid
    grid = Grid(cell_size=1.0, 
                boundary=(bound[1], bound[3]), 
                neighbor_radius=__DX * __H_FAC * 0.9, 
                particle_num=particle_num, 
                particle_pos=world.particle_position)
    
    # init sph model
    base_sph = BaseSPH(boundary=world_bound, h_fac=__H_FAC)
    # sph_model = WCSPH(particle_num, base_sph_model=base_sph, dt=0.0001)
    sph_model = PCISPH(particle_num, base_sph_model=base_sph, dt=0.001)
    # sph_model = IISPH(particle_num, base_sph_model=base_sph, dt=0.001)

    # bind data
    gen.applyToWorld(world)
    sph_model.bindData(world.particle_position, world.particle_velocity, 
                        world.particle_density, world.particle_is_fluid, 
                        grid.particle_num_neighbors, grid.particle_neighbors)
    
    # init simulator
    simulator = Simulator(world, grid, sph_model)
    
    # set colors for rendering
    colors = R.liquidBoundToColors(world.particle_is_fluid)
    
    # update model and render it
    frame = 0
    
    # parameters for saving frames
    save_frame_rate = 30
    save_time_total = 5 # in seconds
    frame_interval = int(1 / (save_frame_rate * sph_model.dt))
    
    save_count = 0
    while R.GUI.running:
        pos = R.taichiToRenderPos(world.particle_position)
        R.renderParticles(pos, colors)
        R.GUI.show() # comment this if you want to save frames
        
        ## Comment R.GUI.show() and 
        ## uncomment these lines to save frames into result folder
        #
        # if frame % frame_interval == 0:
        #     save_count += 1
        #     R.GUI.show(f'result/{save_count}.png')
        #     if save_count / save_frame_rate >= save_time_total:
        #         break

        simulator.update()
        frame += 1

if __name__ == '__main__':
    main()
