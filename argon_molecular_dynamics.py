import numpy as np
import matplotlib.pyplot as plt
import argparse

# =============================================================================
# variables 
# =============================================================================
density = 0.3
temperature = 3.0 
cells_per_axis = 3

timestep = 1e-2
simulation_time = 1500

velocity_relaxation_cutoff_time = 500
velocity_relaxation_frequency = 100
sample_frequency = 50
correlation_function_bins = 100

time_slice_start = velocity_relaxation_cutoff_time
time_slice_end = simulation_time


# =============================================================================
# Variables through command prompt
# =============================================================================

parser = argparse.ArgumentParser(description='Numerical simulation of argon atoms.')
parser.add_argument('-density', default=density, 
                    help='Density of the system.', type=float)
parser.add_argument('-temperature', default=temperature,
                    help='Temperature of the system.', type=float)
parser.add_argument('-cells_per_axis', default=cells_per_axis,
                    help='Number of cells per axis. This will result in 4*(cells_per_axis)**3 particles.', type=int)

parser.add_argument('-timestep', default=timestep, 
                    help='Timestep size used for simulation.', type=float)
parser.add_argument('-simulation_time', default=simulation_time,
                    help='Total number of timesteps taken.', type=int)


parser.add_argument('-velocity_relaxation_cutoff_time', default=velocity_relaxation_cutoff_time,
                    help='At what timestep the velocity relaxation method is disabled.', type=int)
parser.add_argument('-velocity_relaxation_frequency', default=velocity_relaxation_frequency,
                    help='How frequent (in timesteps) velocity relaxation is applied.', type=int)
parser.add_argument('-sample_frequency', default=sample_frequency,
                    help='How frequent (in timesteps) the state of the system is sampled for calculations (after velocity relaxation is disabled).', type=int)
parser.add_argument('-correlation_function_bins', default=correlation_function_bins,
                    help='Number of intervals used for the pair correlation function.', type=int)

parser.add_argument('-time_slice_start', default=time_slice_start,
                    help='Starting timestep of time slice of the data. Used for specifying what data is used in calculations and plots. velocity_relaxation_time_cutoff by default.', type=int)
parser.add_argument('-time_slice_end', default=time_slice_end,
                    help='Starting timestep of time slice of the data. Used for specifying what data is used in calculations and plots. simulation_time by default.', type=int)

parser.add_argument('-no_pressure', action='store_true', help='Disable pressure calculation.')
parser.add_argument('-no_correlation', action='store_true', help='Disable pair correlation function calculation.')
parser.add_argument('-no_energy_plot', action='store_true', help='Disable plotting of Energies.')
parser.add_argument('-save_data', action='store_true', help='Save file containing system data.')

args = parser.parse_args()

density = args.density
temperature = args.temperature
cells_per_axis = args.cells_per_axis

timestep = args.timestep
simulation_time = args.simulation_time
time_slice_start = args.time_slice_start
time_slice_end = args.time_slice_end

velocity_relaxation_cutoff_time = args.velocity_relaxation_cutoff_time
velocity_relaxation_frequency = args.velocity_relaxation_frequency
sample_frequency = args.sample_frequency
correlation_function_bins = args.correlation_function_bins

# =============================================================================
# functions
# =============================================================================

def force(position, length):
    """
    Returns an Nx3 array containing the three-dimensional force vectors per particle.
    The forces are calculated using a dimensionless Lenard-Jones potential.
    The distances between particles are calculated using the minimal image convention.

    Parameters
    ----------
    position : Nx3 array
        The three-dimensional coordinates of the particles in the system.
    length : float
        The length of the system containing the particles.

    Returns
    -------
    forces : Nx3 ndarray
        Array of three-dimensional force vector per particle.
    
    """    
    forces = np.zeros(np.shape(position))
    for i in range(len(position)):
        
        # Create axis specific distances of the ith particle with respect to all other particles.
        # Double nested list comprehesion sums first over the axes, then over all (other) particles.
        x_distances, y_distances, z_distances = [np.asarray([(
            position[i,axis] - other_particle_position + length/2)%length - length/2 
                for other_particle_position in position[:,axis]]) 
                for axis in range(3)]
        
        distances = np.sqrt(x_distances**2 + y_distances**2 + z_distances**2)
        
        # Remove the 0.0 value (distance to itself).
        pair_distances = distances[distances != 0.0]

        # Calculate forces using the derivative of the dimensionless potential with a minus sign in front.
        x_forces, y_forces, z_forces = -1*4*(6*(1/pair_distances)**7 - 12*(1/pair_distances)**13)*[x_distances[distances != 0.0], y_distances[distances != 0.0], z_distances[distances != 0.0]]/pair_distances
        forces[i] = np.asarray([np.sum(x_forces), np.sum(y_forces), np.sum(z_forces)])

    return forces

def potential_energy(position, length):
    """
    Returns the potential energy of the entire system.
    The potential energy is calculated using a dimensionless Lenard-Jones potential.
    The distances between particles are calculated using the minimal image convention.

    Parameters
    ----------
    position : Nx3 array
        The three-dimensional coordinates of the particles in the system.
    length : float
        Length of the system containing the particles.

    Returns
    -------
    energy : float
        The potential energy of the system
    
    """  
    
    energy = 0.0
    for i in range(len(position)):

        # Create axis specific distances of the ith particle with respect to all other particles.
        # Double nested list comprehesion sums first over the axes, then over all (other) particles.
        x_distances, y_distances, z_distances = [np.asarray([(
            position[i,axis]-other_particle_position + length/2)%length - length/2 
                for other_particle_position in position[:,axis]]) 
                for axis in range(3)]
        
        distances = np.sqrt(x_distances**2 + y_distances**2 + z_distances**2)
        
        # Remove the 0.0 value (distance to itself).
        pair_distances = distances[distances != 0.0]
        
        # Calculate the energy using the dimensionless potential.
        # Factor 0.5 corrects the double counting of the interaction energy.
        energy += 0.5*np.sum(4*( (1/pair_distances)**12 - (1/pair_distances)**6 ))
        
    return energy

def system_new_values(position, velocity, length, timestep):
    """
    Returns the positions and velocities of the system after numerical integration of the equations of motion for one timestep.
    The integration makes use of the Velocity Verlet algorithm.
    The forces acting on the particles are caused by a dimensionless Lenard-Jones potential.
    This calculation makes use of periodic boundary conditions.

    Parameters
    ----------
    position : Nx3 array
        The three-dimensional coordinates of the particles in the system.
    velocity : Nx3 array
        The three-dimensional velocity vectors of the particles in the system.
    length : float
        The length of the system containing the particles.
    timestep : float
        The timestep used for the numerical integration.

    Returns
    -------
    new_position : Nx3 ndarray
        The three-dimensional coordinates of N particles of the updated system.
    new_velocity : Nx3 ndarray
        The three-dimensional velocity vectors of N particles of the updated system.
    
    """  
    
    force_position = force(position, length)
    # New positions and velocities are calculated using the Velocity Verlet algorithm with periodic boundary conditions.
    # The list comprehension sums over the axes.
    new_position = np.transpose(
        np.asarray([
        (position[:,axis] + velocity[:,axis]*timestep + 0.5*(timestep**2)*force_position[:,axis])%length 
        for axis in range(3)])
        )

    new_velocity = velocity + 0.5*timestep*(force(new_position, length) + force_position)
    
    return new_position, new_velocity

def system_starting_values(density, temperature, cells_per_axis):
    """
    Returns positions and velocities for a system with a FCC lattice.
    The velocities are randomly generated using a gaussian distribution based on the temperature.
    The spacing of the particles are determined by the density.
    The integration makes use of the Velocity Verlet algorithm.

    Parameters
    ----------
    density : float
        The density of the system.
    temperature : float
        The temperature of the system.
    cells_per_axis : int
        The size of the system expressed in number of cells in one direction. Each cell contains 4 particles.

    Returns
    -------
    positions : ndarray
        The three-dimensional coordinates of the particles in the system.
    velocities : ndarray
        The three-dimensional velocity vectors of the particles in the system.
    number_of_particles : int
        The number of particles in the system. This is equal to 4*(cells_per_axis)**3
    length : float
        The lenght of the system containing the particles.
    
    """  
    number_of_particles = 4*cells_per_axis**3
    length = (number_of_particles/density)**(1/3)
    cell_length = length/cells_per_axis
    cell = 0.5*np.asarray([[0, 0, 0], [cell_length, 0, cell_length], [cell_length, cell_length, 0], [0, cell_length, cell_length]])

    # The positions are created by generating unit cells at a specified location using a displacement vector.
    # The triple nested list comprehension sums over all possible displacement vectors in three dimensions.
    positions = [[[cell + cell_length*np.asarray([i, j, k]) for i in range(cells_per_axis)] for j in range(cells_per_axis)] for k in range(cells_per_axis)]
    positions = np.reshape(positions, (number_of_particles, 3))
    
    velocities = np.random.normal(scale=np.sqrt(temperature), size=(number_of_particles,3))
    
    return positions, velocities, number_of_particles, length

def pairwise_distances(position, length, number_of_particles):
    """
    Returns a 1D array containing all pair-wise distances between particles.

    Parameters
    ----------
    position : Nx3 array
        The three-dimensional coordinates of the particles in the system.
    length : float
        The lenght of the system containing the particles.
    number_of_particles : int
        The number of particles in the system.

    Returns
    -------
    pair_distances : ndarray
        A 1D array containing all pair-wise distances.
        The first N-1 values correspond to all distances to the first particle.
        Here N is the number of particles.
        
    """
    # Index i,j of the array corresponds to the distance between particle i and j.
    # The double nested list comprehension first sums over all particles, then over all distances of other particles to that particle.
    distances = np.asarray([[np.sqrt(
        ((position[n,0]-p[0] + length/2)%length - length/2)**2 +
        ((position[n,1]-p[1] + length/2)%length - length/2)**2 +
        ((position[n,2]-p[2] + length/2)%length - length/2)**2)
        for p in position]
        for n in range(number_of_particles)])
    
    # Remove the 0.0 value to only obtain pair-wise distances.
    pair_distances = distances[distances != 0]
    
    return pair_distances
    
# =============================================================================
# System simulation
# =============================================================================

system_position, system_velocity, number_of_particles, system_length = system_starting_values(density, temperature, cells_per_axis)

system_positions    = np.zeros((simulation_time,number_of_particles,3))
system_velocities   = np.zeros((simulation_time,number_of_particles,3))
system_energies     = np.zeros((simulation_time,3))

# To not use any if statements in a for loop, we split the for loop over the total simulation_time into three parts.
# The double for loop and if statement with for loop simulate the system while velocity relaxation is applied.
# This is done such that the velocity relaxation frequency doens't have to nicely divide the cutoff time.
# The first (double) for loop adjusts the velocities at the specified frequency up to the quotient of the cutoff time.
# The second for loop (with if statement) loops over the remainder. 
# Finally the last for loop continues the simulation for the rest of the time

# t1 in range(quotient of cutoff time divided by the frequency)
for t1 in range(int(velocity_relaxation_cutoff_time/velocity_relaxation_frequency)):
    for t2 in range(velocity_relaxation_frequency):
        
        t = velocity_relaxation_frequency*t1+t2
        system_positions[t]     = system_position
        system_velocities[t]    = system_velocity
        
        system_kinetic_energy          = np.sum(0.5*system_velocity**2)
        system_potential_energy        = potential_energy(system_position, system_length)
        system_energies[t]             = np.asarray([system_kinetic_energy, system_potential_energy, system_kinetic_energy + system_potential_energy])
        
        new_system_position, new_system_velocity = system_new_values(system_position, system_velocity, system_length, timestep)
        system_position     = new_system_position
        system_velocity     = new_system_velocity
        
        
    velocity_correction = np.sqrt((number_of_particles-1)*(3/2)*temperature / system_kinetic_energy)
    system_velocity = velocity_correction*system_velocity

# if there is a remainder, loop over that remainder
if velocity_relaxation_cutoff_time%velocity_relaxation_frequency != 0:
    for t3 in range(velocity_relaxation_cutoff_time%velocity_relaxation_frequency):
        
        t = t3 + velocity_relaxation_frequency*int(np.floor(velocity_relaxation_cutoff_time/velocity_relaxation_frequency))
        system_positions[t]     = system_position
        system_velocities[t]    = system_velocity
        
        system_kinetic_energy          = np.sum(0.5*system_velocity**2)
        system_potential_energy        = potential_energy(system_position, system_length)
        system_energies[t]             = np.asarray([system_kinetic_energy, system_potential_energy, system_kinetic_energy + system_potential_energy])
        
        new_system_position, new_system_velocity = system_new_values(system_position, system_velocity, system_length, timestep)
        system_position     = new_system_position
        system_velocity     = new_system_velocity
        
        
    # velocity relaxation strategy up to a specified cutoff time with a specified frequency
    velocity_correction = np.sqrt((number_of_particles-1)*(3/2)*temperature / system_kinetic_energy)
    system_velocity = velocity_correction*system_velocity       

for t4 in range(simulation_time-velocity_relaxation_cutoff_time):
    
    t = t4+velocity_relaxation_cutoff_time
    system_positions[t]     = system_position
    system_velocities[t]    = system_velocity
    
    system_kinetic_energy          = np.sum(0.5*system_velocity**2)
    system_potential_energy        = potential_energy(system_position, system_length)
    system_energies[t]             = np.asarray([system_kinetic_energy, system_potential_energy, system_kinetic_energy + system_potential_energy])
    

    new_system_position, new_system_velocity = system_new_values(system_position, system_velocity, system_length, timestep)
    system_position     = new_system_position
    system_velocity     = new_system_velocity


system_positions_sliced = system_positions[time_slice_start:time_slice_end]
system_velocities_sliced = system_velocities[time_slice_start:time_slice_end]
system_energies_sliced = system_energies[time_slice_start:time_slice_end]

# =============================================================================
# Pressure and pair-correlation function
# =============================================================================

# check if any calculation has to be done at all
if args.no_pressure and args.no_correlation == True:
    None
else:
    
    pair_distance_distribution_samples  = []
    pressure_double_sum_samples         = []
    
    # same trick as before except different frequency and time interval
    for t1 in range(int((simulation_time-velocity_relaxation_cutoff_time)/sample_frequency)):
        
            t = sample_frequency*t1+velocity_relaxation_cutoff_time
            pair_distances = pairwise_distances(system_positions[t], system_length, number_of_particles)
            
            # Only take into account particles up to L/2 distance apart.
            # This is because beyond this distance the particles aren't found in a uniform shell anymore.
            # This is caused because we simulate in a box, which is not spherically symmetric
            
            max_radius = system_length/2
            
            bin_width = max_radius/correlation_function_bins
            
            # The distribution counts all pair-wise distances that fall inside a specific region.
            # To count this number we apply two conditions on the pair_distances using masks.
            # The object we create is array(pair_distances[lower_bound])[upper_bound].
            # Then we count the length of the array, since it contains all values within the wanted region.
            # The list comprehension sums over the different regions and the factor 0.5 avoids double counting.
            distribution = 0.5*np.array([len(np.array(pair_distances[pair_distances >= bin_index*bin_width])[np.array(pair_distances[pair_distances >= bin_index*bin_width]) < (bin_index+1)*bin_width])
                         for bin_index in range(int(correlation_function_bins))])
            
            pair_distance_distribution_samples.append(
                distribution
                )
            pressure_double_sum_samples.append(
                # This specific value is the double sum component of the formula to calculate the pressure by Verlet.
                0.5*np.sum(0.5*pair_distances*4*(6*(1/pair_distances)**7 - 12*(1/pair_distances)**13))
                )
    
    pressure_double_sum_samples         = np.array(pressure_double_sum_samples)
    pair_distance_distribution_samples  = np.array(pair_distance_distribution_samples)

# =============================================================================
# Pressure calc
# =============================================================================

if args.no_pressure == False:
    pressure = temperature*density*(1 - np.average(pressure_double_sum_samples) / (3*number_of_particles*temperature))
    pressure_std = np.abs(-1*density*np.std(pressure_double_sum_samples) / (3*number_of_particles))
    
    print(f"Pressure = {pressure} +- {pressure_std}")    

# =============================================================================
# pair correlation function
# =============================================================================

if args.no_correlation == False:

    max_distance_particles = max_radius    

    pair_distance_distribution_radii    = np.linspace(bin_width/2, max_distance_particles - bin_width/2, int(correlation_function_bins))
    bin_edges                           = np.linspace(0, max_distance_particles, int(correlation_function_bins)+1)
    pair_distance_distribution_avg      = np.average(pair_distance_distribution_samples, 0)
    pair_distance_distribution_std      = np.std(pair_distance_distribution_samples, 0)
    
    pair_correlation_function           = 2*system_length**3*pair_distance_distribution_avg/(number_of_particles*(number_of_particles-1)*4*np.pi*pair_distance_distribution_radii**2*bin_width)
    pair_correlation_func_error         = np.abs(2*system_length**3*pair_distance_distribution_std/(number_of_particles*(number_of_particles-1)*4*np.pi*pair_distance_distribution_radii**2*bin_width))
    
    plt.stairs(pair_correlation_function, bin_edges)
    
    plt.title('Pair correlation function')
    plt.xlabel('Distance between particles')
    plt.ylabel('Correlation function')  

    plt.show()
    plt.close()

# =============================================================================
# Energy plots
# =============================================================================

if args.no_energy_plot == False:
    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1)
    
    ax1.set_title('Energy plots')
    ax1.plot(np.array(range(time_slice_end-time_slice_start)), system_energies_sliced[:,0], label='$E_{kin}$')
    ax1.plot(np.array(range(time_slice_end-time_slice_start)), system_energies_sliced[:,1], label='$E_{pot}$')
    ax2.plot(np.array(range(time_slice_end-time_slice_start)), system_energies_sliced[:,2], label='$E_{tot}$')
    
    ax1.legend()
    ax1.set_ylabel('Energy')
    
    ax2.legend()
    ax2.set_ylabel('Energy')
    
    plt.xlabel('time')
    plt.show()
    plt.close()

# =============================================================================
# Export data
# =============================================================================

if args.save_data == True:
    exported_data = {
        'density': density,
        'temperature': temperature,
        'positions': system_positions,
        'velocities': system_velocities,
        'energies': system_energies,
        }
    
    if args.no_pressure == False:
        exported_data["pressure"] = pressure
        exported_data["pressure_err"] = pressure_std
        
    if args.no_correlation == False:
        exported_data["correlation_func"] = pair_correlation_function
        exported_data["correlation_func_err"] = pair_correlation_func_error

    np.savez(f"argon_sim_den{density}_temp{temperature}", **exported_data )

# Example of how to read the file:
    
# data=np.load(f"cpa_data/argon_sim_den{density}_temp{temperature}_{data_index}.npz")
# print(data.files)

