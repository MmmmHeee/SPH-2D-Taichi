'''
Major part of SPH implementation
WCSPH and PCISPH are currently implmented
Currently no CFL condition limitation

Reference Paper: 
1. Weakly compressible SPH for free surface flows(WCSPH) 
    Markus Becker   Matthias Teschner
2. Predictive-Corrective Incompressible SPH(PCISPH) 
    B. Solenthaler  R. Pajarola
'''

import taichi as ti
import numpy as np

import math
import smooth_kernel as sk

@ti.data_oriented
class BaseSPH(object):
    # The class is reponsible for particle status update
    # it is bounded with the particle positions and velocities reference

    def __init__(self, 
                boundary = (0, 10, 0, 10),
                dx = 0.1, 
                h_fac = 3,
                rho_0 = 1000.0,
                g = -9.8, 
                alpha = 0.5, 
                c_s = 100.0,
                **kwargs
                ):

        self.boundary = boundary
        self.dx = dx # particle distance at reference density
        self.h_fac = h_fac
        self.rho_0 = rho_0 # reference density
        self.g = g # gravity (-9.8)
        self.alpha = alpha # viscosity constant        
        self.c_s = c_s # sound speed in liquid

        self.dh = dx * h_fac # kernel radius

        self.left_bound = boundary[0]
        self.right_bound = boundary[1]
        self.bottom_bound = boundary[2]
        self.top_bound = boundary[3]

        self.m = 1

        # The kernel for differnet steps, check smooth_kernel.py
        self.densityKernel = sk.cubicKernel
        self.densityGrad = sk.cubicGrad
        self.pressureGrad = sk.cubicGrad
        self.viscosityGrad = sk.cubicGrad
        self.factorGrad = sk.cubicGrad

        # these data are reference, would be bounded using bindData()
        self.particle_position = None
        self.particle_velocity = None
        self.particle_density = None
        self.particle_is_fluid = None

        self.particle_num_neighbors = None
        self.particle_neighbors = None

    # bind refernce
    def bindData(self, pos, vel, density, is_fluid, num_neighbors, neighbors):
        self.particle_position = pos
        self.particle_velocity = vel
        self.particle_density = density
        self.particle_is_fluid = is_fluid

        self.particle_num_neighbors = num_neighbors
        self.particle_neighbors = neighbors

    @ti.func
    def rho_sum(self, x_ab):
        # WCSPH equation (4)
        # Weakly compressible SPH for free surface flows
        # calculate density by sum up neighbors
        return self.m * self.densityKernel(x_ab, self.dh)

    @ti.func
    def rho_dt(self, v_ab, x_ab):
        # WCSPH equation (5)
        # update density by derivation
        return self.m * v_ab.dot(self.densityGrad(x_ab, self.dh))

    @ti.func
    def vel_dt_from_pressure(self, Pa, Pb, rho_a, rho_b, x_ab):
        # WCSPH equation (6) without gravity
        # Compute the pressure force contribution, Symmetric Formula
        res = ti.Vector([0.0, 0.0], dt=float)
        res = -self.m * (Pa / rho_a ** 2 + Pb / rho_b ** 2) * self.pressureGrad(x_ab, self.dh)
        return res

    @ti.func
    def vel_dt_from_viscosity(self, rho_a, rho_b, v_ab, x_ab):
        # WCSPH equation (10)
        # Compute the viscosity force contribution, artificial viscosity
        res = ti.Vector([0.0, 0.0], dt=float)
        v_dot_x = v_ab.dot(x_ab)
        if v_dot_x < 0:
            # Artifical viscosity
            mu = 2.0 * self.alpha * self.dh * self.c_s / (rho_a + rho_b)
            PI_ab = - mu * (v_dot_x / (x_ab.norm()**2 + 0.01 * self.dh**2))
            res = - self.m * PI_ab * self.viscosityGrad(x_ab, self.dh)
        return res

    @ti.kernel
    def calcMass(self) -> float:
        # calculate mass so that it fits the radius of kernel and reference density
        density_sum = 0.
        half_range = self.dh / self.dx

        # filled neighbor with particle distance self.dx
        for i_x in range(-half_range, half_range + 1):
            for i_y in range(-half_range, half_range + 1):
                r = ti.Vector([i_x, i_y], dt=float) * self.dx
                density_sum += self.rho_sum(r) / self.m

        return ti.cast(self.rho_0 / density_sum, float) # reference to WCSPH equation (4)

    @ti.kernel
    def enforcingBoundary(self):
        # should keep same with PCISPH.enforcingBoundaryPrediction
        dist_epsilon = 1e-3 # avoid particles from sticking into each other
        speed_epsilon = 0
        for p_i in self.particle_position:
            if self.particle_is_fluid[p_i] == 1:
                pos = self.particle_position[p_i]
                if pos[0] < self.left_bound:
                    self.particle_position[p_i][0] = self.left_bound + dist_epsilon * ti.random()
                    self.particle_velocity[p_i][0] *= -speed_epsilon
                elif pos[0] > self.right_bound:
                    self.particle_position[p_i][0] = self.right_bound - dist_epsilon * ti.random()
                    self.particle_velocity[p_i][0] *= -speed_epsilon
                if pos[1] < self.bottom_bound:
                    self.particle_position[p_i][1] = self.bottom_bound + dist_epsilon * ti.random()
                    self.particle_velocity[p_i][1] *= -speed_epsilon
                elif pos[1] > self.top_bound:
                    self.particle_position[p_i][1] = self.top_bound - dist_epsilon * ti.random()
                    self.particle_velocity[p_i][1] *= -speed_epsilon

    def step(self):
        # an interface for inheritance 
        # implemented in class WCSPH and PCISPH
        pass

## TODO: add adaptive timestep based on CFL condition
@ti.data_oriented
class WCSPH(BaseSPH):
    def __init__(self, particle_num,
                base_sph_model = BaseSPH(),
                gamma = 7,
                dt = 0.0002
                ):

        super(WCSPH, self).__init__(**base_sph_model.__dict__)
        
        # tait function parameters
        self.gamma = gamma # ref to WCSPH equation (7)
        self.B = self.rho_0 * self.c_s**2 / gamma # ref to WCSPH equation (7)

        self.dt = dt

        self.particle_pressure = ti.var(dt = float) # save particle pressure
        self.d_density = ti.var(dt = float) # save the change of density 
        self.d_velocity = ti.Vector(2, dt=float) # and velocity

        ti.root.dense(ti.i, particle_num).place(self.particle_pressure)
        ti.root.dense(ti.i, particle_num).place(self.d_density, self.d_velocity)

        self.m = self.calcMass()

    @ti.func
    def tait_function(self, rho):
        # tait equation, ref to WCSPH equation (7)
        return self.B * ((rho / self.rho_0)**self.gamma - 1.0)        

    @ti.kernel
    def calcDensityVelocityChange(self):
        # ref to WCSPH paper or PCISPH paper algorithm 1
        for pa_idx in self.particle_position:
            if self.particle_is_fluid[pa_idx] == 1:

                # get status for particle A
                pos_a = self.particle_position[pa_idx] # position
                vel_a = self.particle_velocity[pa_idx] # velocity
                rho_a = self.particle_density[pa_idx] # density
                P_a = self.particle_pressure[pa_idx] # pressure

                d_v = ti.Vector([0.0, 0.0], dt=float)
                d_rho = 0
                for nb_idx in range(self.particle_num_neighbors[pa_idx]):
                    pb_idx = self.particle_neighbors[pa_idx, nb_idx]

                    if self.particle_is_fluid[pb_idx] == 1:
                        vel_b = self.particle_velocity[pb_idx]
                        v_ab = vel_a - vel_b

                        pos_b = self.particle_position[pb_idx]
                        x_ab = pos_a - pos_b

                        rho_b = self.particle_density[pb_idx]
                        P_b = self.particle_pressure[pb_idx]
                        d_v += self.vel_dt_from_pressure(P_a, P_b, rho_a, rho_b, x_ab)

                        # Compute Density change
                        d_rho += self.rho_dt(v_ab, x_ab)
                        # apply Viscosity force
                        d_v += self.vel_dt_from_viscosity(rho_a, rho_b, v_ab, x_ab)

                # Apply gravity
                if self.particle_is_fluid[pa_idx] == 1:
                    d_v += ti.Vector([0.0, self.g])

                self.d_velocity[pa_idx] = d_v
                self.d_density[pa_idx] = d_rho

    @ti.kernel
    def updateStatus(self):
        # Forward Euler
        for p_i in self.particle_position:
            if self.particle_is_fluid[p_i] == 1:
                self.particle_velocity[p_i] += self.dt * self.d_velocity[p_i]
                self.particle_position[p_i] += self.dt * self.particle_velocity[p_i]
                self.particle_density[p_i] += self.dt * self.d_density[p_i]
                self.particle_pressure[p_i] = self.tait_function(self.particle_density[p_i])

    def step(self):
        # ref to WCSPH paper or PCISPH paper algorithm 1
        self.calcDensityVelocityChange() # compute density and velocity change
        self.updateStatus() # update velocity, positions, density ..

@ti.data_oriented
class PCISPH(BaseSPH):

    def __init__(self, particle_num,
                base_sph_model = BaseSPH(),
                fluctuation_threshold = 0.001,
                least_iteration = 5,
                max_iteration = 1e4,
                dt = 0.0008,
                ):

        super(PCISPH, self).__init__(**base_sph_model.__dict__)
        
        self.fluctuation_threshold = fluctuation_threshold # fluctuation limit (max_density_error < 0.01)
        self.least_iteration = least_iteration # the minimum times of iteration
        self.max_iteration = max_iteration # the maximum times of iteration
        self.dt = dt # timestep

        self.predicted_position = ti.Vector.field(2, dtype = float)
        self.predicted_velocity = ti.Vector.field(2, dtype = float)
        self.predicted_density = ti.field(dtype = float)
        self.corrected_pressure = ti.field(dtype = float)
        
        self.dv_without_pressure = ti.Vector.field(2, dtype=float)
        self.dv_pressure = ti.Vector.field(2, dtype=float)

        ti.root.dense(ti.i, particle_num).place(self.predicted_position, self.predicted_velocity, self.predicted_density,
                                                self.corrected_pressure, self.dv_without_pressure, self.dv_pressure)
        self.max_density_error = ti.field(dtype=float, shape=())
        self.m = self.calcMass()
        self.delta = self.calcScalingFactor()
        
    @ti.kernel
    def calcScalingFactor(self) -> float:
        # ref to PCISPH equation(8)
        # calculate gradient dot and sum in filled neighbors
        grad_sum = ti.Vector([0.0, 0.0], dt=float)
        grad_dot_sum = 0.0
        half_range = self.dh / self.dx

        # filled neighbor with particle distance self.dx
        for i_x in range(-half_range, half_range + 1):
            for i_y in range(-half_range, half_range + 1):
                r = ti.Vector([i_x, i_y], dt=float) * ti.cast(self.dx, float)
                r_len = r.norm()
                if r_len <= self.dh and r_len >= 1e-5:
                    grad = self.factorGrad(r, ti.cast(self.dh, float))
                    grad_sum += grad
                    grad_dot_sum += grad.dot(grad)
        beta = 2 * (self.dt * self.m / self.rho_0)**2
        return -1.0 / (beta * (grad_sum.dot(grad_sum) - grad_dot_sum))

    @ti.kernel
    def calcVelocityChangeWithoutPressure(self):
        for pa_idx in self.particle_position:
            if self.particle_is_fluid[pa_idx] == 1:
                # get status for particle A
                pos_a = self.particle_position[pa_idx] # position
                vel_a = self.particle_velocity[pa_idx] # velocity
                rho_a = self.particle_density[pa_idx] # density

                d_v = ti.Vector([0.0, 0.0], dt=float)
                for nb_idx in range(self.particle_num_neighbors[pa_idx]):
                    pb_idx = self.particle_neighbors[pa_idx, nb_idx]
                    if self.particle_is_fluid[pb_idx] == 1:
                        # get status for particle B
                        pos_b = self.particle_position[pb_idx]
                        vel_b = self.particle_velocity[pb_idx]
                        rho_b = self.particle_density[pb_idx]

                        x_ab = pos_a - pos_b
                        v_ab = vel_a - vel_b
                        
                        # apply viscosity
                        d_v += self.vel_dt_from_viscosity(rho_a, rho_b, v_ab, x_ab)

                d_v += ti.Vector([0.0, self.g])
                self.dv_without_pressure[pa_idx] = d_v

    @ti.kernel
    def calcVelocityChangeForCorrectedPressure(self):
        for pa_idx in self.particle_position:
            if self.particle_is_fluid[pa_idx] == 1:
                
                # get status for particle A
                pos_a = self.particle_position[pa_idx] # position
                rho_a = self.particle_density[pa_idx] # density
                P_a = self.corrected_pressure[pa_idx] # corrected pressure for prediction

                d_v = ti.Vector([0.0, 0.0], dt=float)
                for nb_idx in range(self.particle_num_neighbors[pa_idx]):
                    pb_idx = self.particle_neighbors[pa_idx, nb_idx]                       
                    if self.particle_is_fluid[nb_idx]:
                        # get status for particle B
                        pos_b = self.particle_position[pb_idx]
                        rho_b = self.particle_density[pb_idx]
                        P_b = self.corrected_pressure[pb_idx]

                        x_ab = pos_a - pos_b
                        # apply corrected pressure force for prediction
                        d_v += self.vel_dt_from_pressure(P_a, P_b, rho_a, rho_b, x_ab)

                self.dv_pressure[pa_idx] = d_v

    @ti.kernel
    def predictPositionAndVelocity(self):
        for p_i in self.particle_is_fluid:
            if self.particle_is_fluid[p_i] == 1:
                self.predicted_velocity[p_i] = self.particle_velocity[p_i] + \
                                                (self.dv_pressure[p_i] + \
                                                self.dv_without_pressure[p_i]) * self.dt
                self.predicted_position[p_i] = self.particle_position[p_i] + self.predicted_velocity[p_i] * self.dt

    @ti.kernel
    def enforcingBoundaryPrediction(self):
        # keep same to BaseSPH.enforcingBoundary
        # applied to predicted particle data
        dist_epsilon = 1e-3
        speed_epsilon = 0
        for p_i in self.predicted_position:
            if self.particle_is_fluid[p_i] == 1:
                pos = self.particle_position[p_i]
                if pos[0] < self.left_bound:
                    self.predicted_position[p_i][0] = self.left_bound + dist_epsilon * ti.random()
                    self.predicted_velocity[p_i][0] *= speed_epsilon
                elif pos[0] > self.right_bound:
                    self.predicted_position[p_i][0] = self.right_bound - dist_epsilon * ti.random()
                    self.predicted_velocity[p_i][0] *= speed_epsilon
                if pos[1] < self.bottom_bound:
                    self.predicted_position[p_i][1] = self.bottom_bound + dist_epsilon * ti.random()
                    self.predicted_velocity[p_i][1] *= speed_epsilon
                elif pos[1] > self.top_bound:
                    self.predicted_position[p_i][1] = self.top_bound - dist_epsilon * ti.random()
                    self.predicted_velocity[p_i][1] *= speed_epsilon

    @ti.kernel
    def updateCorrectedPressure(self):
        # predict density, max density error, correction pressure
        for pa_idx in self.predicted_position:
            if self.particle_is_fluid[pa_idx] == 1:
                pos_a = self.predicted_position[pa_idx] # position
                vel_a = self.predicted_velocity[pa_idx] # velocity

                # using two different method to predict density and 
                # choose the one which is closer to reference density
                predicted_density = self.particle_density[pa_idx]

                for nb_idx in range(self.particle_num_neighbors[pa_idx]):
                    pb_idx = self.particle_neighbors[pa_idx, nb_idx]
                    
                    pos_b = self.predicted_position[pb_idx]
                    vel_b = self.predicted_velocity[pb_idx]

                    x_ab = pos_a - pos_b
                    v_ab = vel_a - vel_b

                    if self.particle_is_fluid[nb_idx] == 1:
                        predicted_density += self.rho_dt(v_ab, x_ab) * self.dt

                # only handle positive error (thus incompressible)
                density_error = max(predicted_density - self.rho_0, 0)

                self.predicted_density[pa_idx] = predicted_density
                self.corrected_pressure[pa_idx] += self.delta * density_error # ref to PCISPH equation (9),(10)

                err = abs(density_error)

                # update max density error
                self.max_density_error[None] = ti.atomic_max(err, self.max_density_error[None])
            
    @ti.kernel
    def updateStatus(self):
        # velocity, position and density
        for pa_idx in self.particle_is_fluid:
            if self.particle_is_fluid[pa_idx] == 1:
                self.particle_velocity[pa_idx] += (self.dv_pressure[pa_idx] + \
                                                self.dv_without_pressure[pa_idx]) * self.dt
                self.particle_position[pa_idx] += self.particle_velocity[pa_idx] * self.dt
                
                pos_a = self.particle_position[pa_idx]
                # vel_a = self.particle_velocity[pa_idx]

                ## the deltatime update method is not that stable
                # density = self.particle_density[pa_idx]

                # use summing method instead, however causing boundary problem due to lack of particles
                density = 0.

                for nb_idx in range(self.particle_num_neighbors[pa_idx]):
                    pb_idx = self.particle_neighbors[pa_idx, nb_idx]
                    
                    # if self.particle_is_fluid[pb_idx] == 1:
                    pos_b = self.particle_position[pb_idx]
                    vel_b = self.particle_velocity[pb_idx]

                    x_ab = pos_a - pos_b
                    # v_ab = vel_a - vel_b

                    # if self.particle_is_fluid[nb_idx] == 1:
                    #     density += self.rho_dt(v_ab, x_ab) * self.dt
                    density += self.rho_sum(x_ab)

                # update predicted density and corrected_pressure
                self.particle_density[pa_idx] = density

    def step(self):
        # ref to PCISPH algorithm 2
        self.calcVelocityChangeWithoutPressure()
        self.corrected_pressure.fill(0)
        self.dv_pressure.fill(0)

        # Start density prediction-correction iteration process
        it = 0 # number of iteration        
        self.max_density_error[None] = 0.
        while (it < self.least_iteration or self.max_density_error[None] > self.fluctuation_threshold):
            self.predictPositionAndVelocity()
            self.enforcingBoundaryPrediction()
            self.updateCorrectedPressure()
            self.calcVelocityChangeForCorrectedPressure()
            
            it += 1
            if it > self.max_iteration:
                print("After" + str(self.max_iteration) + "iterations still cannot converge")
                print("Error:", self.max_density_error[None])
                break

        # Compute new velocity, position and density
        self.updateStatus()