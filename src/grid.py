'''
Grid structure implemented to speed up neighbor searching
Ref to pbf2d.py of taichi example implemented by Ye Kuang (k-ye)
'''
import taichi as ti
import numpy as np

@ti.data_oriented
class Grid(object):
    def __init__(self, cell_size, boundary, neighbor_radius, particle_num, particle_pos: ti.Vector):
        self.MAX_NUM_PARTICLES_PER_CELL = 500
        self.MAX_NUM_NEIGHBORS = 500

        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)

        self.cell_size = cell_size
        self.grid_dim = np.ceil( np.array(boundary) / cell_size).astype(int)
        self.neighbor_radius = neighbor_radius

        grid_snode = ti.root.dense(ti.ij, self.grid_dim)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.k, self.MAX_NUM_PARTICLES_PER_CELL).place(
            self.grid2particles)

        nb_node = ti.root.dense(ti.i, particle_num)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.MAX_NUM_NEIGHBORS).place(
            self.particle_neighbors)
    
        self.particle_positions = particle_pos # a reference, not copy

    @ti.func
    def getCellIndex(self, pos) -> ti.Vector:
        return (pos / self.cell_size).cast(int)

    @ti.kernel
    def allocateParticles(self):
        # allocate particles to grid
        for p_i in self.particle_positions:
            cell_idx = self.getCellIndex(self.particle_positions[p_i]) # cell index

            # next position
            offs = self.grid_num_particles[cell_idx].atomic_add(1)
            self.grid2particles[cell_idx, offs] = p_i

    @ti.func
    def is_in_grid(self, c) -> bool:
        return 0 <= c[0] and c[0] < self.grid_dim[0] and 0 <= c[1] and c[
            1] < self.grid_dim[1]

    @ti.kernel
    def searchNeighbors(self):
        for p_i in self.particle_positions:
            pos_i = self.particle_positions[p_i]
            nb_i = 0 # count the number of neighbors

            cell = self.getCellIndex(self.particle_positions[p_i])

            # check all particles in current and neighbour cell
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                cell_to_check = cell + offs
                if self.is_in_grid(cell_to_check) == 1:
                    for j in range(self.grid_num_particles[cell_to_check]):
                        p_j = self.grid2particles[cell_to_check, j]

                        # check the distance of the particles
                        if nb_i < self.MAX_NUM_NEIGHBORS and p_j != p_i and (
                                pos_i - self.particle_positions[p_j]
                        ).norm() < self.neighbor_radius:
                            self.particle_neighbors[p_i, nb_i] = p_j # add neighbor
                            nb_i.atomic_add(1) # add counter
            self.particle_num_neighbors[p_i] = nb_i

    def updateNeighbors(self):
        # clear grid
        self.grid_num_particles.fill(0)
        self.grid2particles.fill(-1)
        self.particle_num_neighbors.fill(0)
        self.particle_neighbors.fill(-1)

        self.allocateParticles()
        self.searchNeighbors()