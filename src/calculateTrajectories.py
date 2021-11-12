import numpy as np
# from src import forces

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
    
    for i in range(len(times)):
        newPositions, newVelocities = updateParticles(masses, positions, velocities, dt)  # update positions and velocities
        allPositions[:,:,i] = newPositions  # store new positions
        allVelocities[:,:,i] = newVelocities  # store new velocities
        positions, velocities = newPositions, newVelocities  # update positions and velocities for next iteration
        print('{:.2f}% done'.format(i/len(times)*100))  
        
    return times, allPositions, allVelocities