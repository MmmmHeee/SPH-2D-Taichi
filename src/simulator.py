from world import World
from grid import Grid
from model import BaseSPH

class Simulator(object):
    def __init__(self, world: World, grid: Grid, sph_model: BaseSPH):
        self.world = world
        self.grid = grid
        self.sph_model = sph_model

    # the main looped step of the simulation
    def update(self):
        self.grid.updateNeighbors()
        self.sph_model.step()
        self.sph_model.enforcingBoundary()
