import numpy as np
from src import forces
from tqdm import tqdm

def updateParticles(masses, positions, velocities, dt):
    """
    Evolve particles in time via leap-frog integrator scheme. This function
    takes masses, positions, velocities, and a time step dt as

    Parameters
    ----------
    masses : np.ndarray
        1-D array containing masses for all particles, in kg
        It has length N, where N is the number of particles.
    positions : np.ndarray
        2-D array containing (x, y, z) positions for all particles.
        Shape is (N, 3) where N is the number of particles.
    velocities : np.ndarray
        2-D array containing (x, y, z) velocities for all particles.
        Shape is (N, 3) where N is the number of particles.
    dt : float
        Evolve system for time dt (in seconds).

    Returns
    -------
    Updated particle positions and particle velocities, each being a 2-D
    array with shape (N, 3), where N is the number of particles.

    """

    startingPositions = np.array(positions)
    startingVelocities = np.array(velocities)

    # how many particles are there?
    nParticles, nDimensions = startingPositions.shape

    # make sure the three input arrays have consistent shapes
    assert(startingVelocities.shape == startingPositions.shape)
    assert(len(masses) == nParticles)

    # calculate net force vectors on all particles, at the starting position
    startingForces = np.array(forces.calculateForceVectors(masses, startingPositions))

    # calculate the acceleration due to gravity, at the starting position
    startingAccelerations = startingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending position
    nudge = startingVelocities*dt + 0.5*startingAccelerations*dt**2  # nudge = v*dt + 1/2*a*dt^2
    endingPositions = startingPositions + nudge

    # calculate net force vectors on all particles, at the ending position
    endingForces = np.array(forces.calculateForceVectors(masses, endingPositions))

    # calculate the acceleration due to gravity, at the ending position
    endingAccelerations = endingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending velocity
    endingVelocities = (startingVelocities +
                        0.5*(endingAccelerations + startingAccelerations)*dt)

    return endingPositions, endingVelocities

def calculateTrajectories(masses, positions, velocities, duration, dt):
    
    """
    Evolves particle's positions velocities in time.
    Starts from initial positions and velocities, takes a step forward in time,
    calculates new positions and velocities, stores these new values in arrays,
    and then repeat that process over and over again.
    
    Parameters
    ----------
    masses: 1D array, with nParticles elements
    positions: 2D array, nParticles × nDimensions elements
    velocities: 2D array, nParticles × nDimensions elements
    duration: the total time to evolve the system (a float, in seconds)
    dt: the size of each time step, (a float, in seconds)
    
    Returns
    -------
    times: 1D array, with nTimes elements
    positions at all times: 3D array, nParticles × nDimensions × nTimes
    velocities at all times: 3D array, nParticles × nDimensions × nTimes
    """
    
    startingPositions = np.array(positions)
    startingVelocities = np.array(velocities)
    
    # how many particles are there?
    nParticles, nDimensions = startingPositions.shape
    
    # make sure the three input arrays have consistent shapes
    assert(startingVelocities.shape == startingPositions.shape)
    assert(len(masses) == nParticles)
    
    times = np.arange(0, duration, dt)
    allPositions = np.zeros((nParticles, nDimensions, len(times)))
    allVelocities = np.zeros((nParticles, nDimensions, len(times)))
    
    print("Calculating trajectories...")
    for i in tqdm(range(len(times))):
        newPositions, newVelocities = updateParticles(masses, positions, velocities, dt)
        allPositions[:,:,i] = newPositions
        allVelocities[:,:,i] = newVelocities
        positions, velocities = newPositions, newVelocities
        # print('{:.2f}% done'.format(i/len(times)*100))
        
    return times, allPositions, allVelocities
