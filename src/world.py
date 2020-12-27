import taichi as ti
import numpy as np
from grid import Grid

# particle world
# save all information of all particles
@ti.data_oriented
class World(object):
    def __init__(self,
                particle_num,
                ):

        self.particle_num = particle_num
        
        self.particle_position = ti.Vector.field(2, dtype=float)
        self.particle_velocity = ti.Vector.field(2, dtype=float)
        self.particle_is_fluid = ti.field(dtype = int)
        self.particle_density = ti.field(dtype = float)

        ti.root.dense(ti.i, self.particle_num).place(
                                        self.particle_position,
                                        self.particle_velocity,
                                        self.particle_is_fluid,
                                        self.particle_density
                                    )        

    def setParticles(self, pos: list, vel: list, is_fluid: list, density: list):
        self.particle_position.from_numpy(np.array(pos, dtype=np.float))
        self.particle_velocity.from_numpy(np.array(vel, dtype=np.float))
        self.particle_is_fluid.from_numpy(np.array(is_fluid, dtype=np.int))
        self.particle_density.from_numpy(np.array(density, dtype=np.float))

# A simple world generator
# can draw cube
class WorldGen:
    def __init__(self, dx):
        
        self.dx = dx

        self.p_pos_list = []
        self.p_is_fluid_list = []
        self.p_density_list = []
        self.p_velocity_list = []

    def add_cube(self,
                 lower_bound: list,
                 upper_bound: list,
                 is_fluid,
                 density,
                 velocity=None):

        for pos_x in np.arange(lower_bound[0], upper_bound[0] + self.dx, self.dx):
            for pos_y in np.arange(lower_bound[1], upper_bound[1] + self.dx, self.dx):
                self.p_pos_list.append([pos_x, pos_y])
                self.p_is_fluid_list.append(is_fluid)
                self.p_density_list.append(density)
                if velocity:
                    self.p_velocity_list.append([velocity[0], velocity[1]])
                else:
                    self.p_velocity_list.append([0, 0])

    def add_cube_random(self,
                        lower_bound: list,
                        upper_bound: list,
                        is_fluid,
                        density,
                        velocity = None):
        for pos_x in np.arange(lower_bound[0], upper_bound[0] + self.dx, self.dx):
            for pos_y in np.arange(lower_bound[1], upper_bound[1] + self.dx, self.dx):
                rand_1 = (np.random.rand() - 0.5) * 0.6
                rand_2 = (np.random.rand() - 0.5) * 0.6

                self.p_pos_list.append([pos_x + rand_1 * self.dx, pos_y + rand_2 * self.dx])
                self.p_is_fluid_list.append(is_fluid)
                self.p_density_list.append(density)
                if velocity:
                    self.p_velocity_list.append([velocity[0], velocity[1]])
                else:
                    self.p_velocity_list.append([0, 0])

    def applyToWorld(self, world):
        world.setParticles(self.p_pos_list, self.p_velocity_list,
                            self.p_is_fluid_list, self.p_density_list)

    def getParticleNum(self) -> int:
        return len(self.p_pos_list)