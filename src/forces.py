import numpy as np
import matplotlib.pyplot as plt

def rescale(vectors, magnitudes):
    return vectors
    n = len(magnitudes)
    for i in range(n):
        omag = magnitude(vectors[i])
        #print(magnitudes[i]/omag)
        vectors[i] = vectors[i]*magnitudes[i]/omag
    return vectors

def forceMagnitude(mi, mj, sep):
    """
    Compute magnitude of gravitational force between two particles.

    Parameters
    ----------
    mi, mj : float
        Particle masses in kg.
    sep : float
        Particle separation (distance between particles) in m.

    Returns
    -------
    force : float
        Gravitational force between particles in N.

    Example
    -------
        Input:
            mEarth = 6.0e24     # kg
            mPerson = 70.0      # kg
            radiusEarth = 6.4e6 # m
            print(magnitudeOfForce(mEarth, mPerson, radiusEarth))
        Output:
            683.935546875
    """
    G = 6.67e-11                # m3 kg-1 s-2
    return G * mi * mj / sep**2 # N

def magnitude(vec):
    """
    Compute magnitude of any vector with an arbitrary number of elements.

    Parameters
    ----------
    vec : numpy array
        Any vector

    Returns
    -------
    magnitude : float
        The magnitude of that vector.

    Example
    -------
        Input:
            print(magnitude(np.array([3.0, 4.0, 0.0])))
        Output:
            5.0
    """
    return np.sqrt(np.sum(vec**2))

def unitDirectionVector(pos_a, pos_b):
    """
    Create unit direction vector from pos_a to pos_b

    Parameters
    ----------
    pos_a, pos_b : two numpy arrays
        Any two vectors

    Returns
    -------
    unit direction vector : one numpy array (same size input vectors)
        The unit direction vector from pos_a toward pos_b

    Example
    -------
        Input:
            someplace = np.array([3.0,2.0,5.0])
            someplaceelse = np.array([1.0, -4.0, 8.0])
            print(unitDirectionVector(someplace, someplaceelse))
        Output:
            [-0.28571429, -0.85714286,  0.42857143]
    """

    # calculate the separation between the two vectors
    separation = pos_b - pos_a

    # divide vector components by vector magnitude to make unit vector
    return separation/magnitude(separation)

def forceVector(mi, mj, pos_i, pos_j):
    """
    Compute gravitational force vector exerted on particle i by particle j.

    Parameters
    ----------
    mi, mj : floats
        Particle masses, in kg.
    pos_i, pos_j : numpy arrays
        Particle positions in cartesian coordinates, in m.

    Returns
    -------
    forceVec : numpy array
        Components of gravitational force vector, in N.

    Example
    -------
        Input:
            mEarth = 6.0e24     # kg
            mPerson = 70.0      # kg
            radiusEarth = 6.4e6 # m
            centerEarth = np.array([0,0,0])
            surfaceEarth = np.array([0,0,1])*radiusEarth
            print(forceVector(mEarth, mPerson, centerEarth, surfaceEarth))

        Output:
            [   0.            0.          683.93554688]


    """

    # compute the magnitude of the distance between positions
    distance = magnitude(pos_i - pos_j)
    # this distance is in meters, because pos_i and pos_j were

    # compute the magnitude of the force
    force = forceMagnitude(mi, mj, distance)
    # the magnitude of the force is in Newtons

    # calculate the unit direction vector of the force
    direction = unitDirectionVector(pos_i, pos_j)
    # this vector is unitless, its magnitude should be 1.0

    return force*direction # a numpy array, with units of Newtons


def calculateForceVector(masses, positions, i, position):
    vector = np.zeros(3)
    N = len(positions)

    # loop over all the particles we need to include in the force sum
    for j in range(N):

        # as long as i and j are not the same...
        if j != i:

            # ...add in the force vector of particle j acting on particle i
            vector += forceVector(masses[i], masses[j], position, positions[j])

    return vector

# define a function to calculate force vectors for all particles
def calculateForceVectors(masses, positions):
    """
    Compute net gravitational force vectors on particles,
    given a list of masses and positions for all of them.

    Parameters
    ----------
    masses : list (or 1D numpy array) of floats
        Particle masses, in kg.
    positions : list (or numpy array) of 3-element numpy arrays
        Particle positions in cartesian coordinates, in meters,
        in the same order as the masses are listed. Each element
        in the list (a single particle's position) should be a
        3-element numpy array, referring to its X, Y, Z position.

    Returns
    -------
    forceVectrs : list of 3-element numpy arrays
        A list containing the net force vectors for each particles.
        Each element in the list is a 3-element numpy array that
        represents the net 3D force acting on a particle, after summing
        over the individual force vectors induced by every other particle.

    Example
    -------
        Input:
            au = 1.496e+11
            masses = [1.0e24, 40.0e24, 50.0e24, 30.0e24, 2.0e24]
            positions = [np.array([ 0.5,  2.6,  0.05])*au,
                         np.array([ 0.8,  9.1,  0.10])*au,
                         np.array([-4.1, -2.4,  0.80])*au,
                         np.array([10.7,  3.7,  0.00])*au,
                         np.array([-2.0, -1.9, -0.40])*au]

            # calculate and print the force vectors for all particles
            forces = calculateForceVectors(masses, positions)

            print('{:>10} | {:>10} | {:>10} | {:>10}'.format('particle', 'Fx', 'Fy', 'Fz'))
            print('{:>10} | {:>10} | {:>10} | {:>10}'.format('(#)', '(N)', '(N)', '(N)'))
            print('-'*49)
            for i in range(len(forces)):
                Fx, Fy, Fz = forces[i]
                print('{:10.0f} | {:10.1e} | {:10.1e} | {:10.1e}'.format(i, Fx, Fy, Fz))

        Output:
              particle |         Fx |         Fy |         Fz
                   (#) |        (N) |        (N) |        (N)
            -------------------------------------------------
                     0 |   -1.3e+15 |    3.8e+14 |    3.5e+14
                     1 |    9.2e+15 |   -5.3e+16 |    1.8e+15
                     2 |    7.5e+16 |    5.4e+16 |   -2.7e+16
                     3 |   -4.2e+16 |    6.4e+15 |    1.1e+15
                     4 |   -4.0e+16 |   -7.5e+15 |    2.4e+16

        """

    # how many particles are there?
    N = len(positions)

    # create an empty list, which we will fill with force vectors
    forcevectors = []

    # loop over particles for which we want the force vector
    for i in range(N):

        # create a force vector with all three elements as zero
        vector = np.zeros(3)

        # loop over all the particles we need to include in the force sum
        for j in range(N):

            # as long as i and j are not the same...
            if j != i:

                # ...add in the force vector of particle j acting on particle i
                vector += forceVector(masses[i], masses[j], positions[i], positions[j])

        # append this force vector into the list of force vectors
        forcevectors.append(vector)

    # return the list of force vectors out of the function
    return forcevectors


def test():
    '''This function tests the force calculations.'''

    print("This test function should produce the following output:")

    print("""
  particle |         Fx |         Fy |         Fz
       (#) |        (N) |        (N) |        (N)
-------------------------------------------------
         0 |   -1.3e+15 |    3.8e+14 |    3.5e+14
         1 |    9.2e+15 |   -5.3e+16 |    1.8e+15
         2 |    7.5e+16 |    5.4e+16 |   -2.7e+16
         3 |   -4.2e+16 |    6.4e+15 |    1.1e+15
         4 |   -4.0e+16 |   -7.5e+15 |    2.4e+16
    """)

    print("Here is what it outputs after running calculateForceVectors:\n")

    au = 1.496e+11
    masses = [1.0e24, 40.0e24, 50.0e24, 30.0e24, 2.0e24]
    positions = [np.array([ 0.5,  2.6,  0.05])*au,
                 np.array([ 0.8,  9.1,  0.10])*au,
                 np.array([-4.1, -2.4,  0.80])*au,
                 np.array([10.7,  3.7,  0.00])*au,
                 np.array([-2.0, -1.9, -0.40])*au]

    # calculate and print the force vectors for all particles
    forces = calculateForceVectors(masses, positions)

    print('{:>10} | {:>10} | {:>10} | {:>10}'.format('particle', 'Fx', 'Fy', 'Fz'))
    print('{:>10} | {:>10} | {:>10} | {:>10}'.format('(#)', '(N)', '(N)', '(N)'))
    print('-'*49)
    for i in range(len(forces)):
        Fx, Fy, Fz = forces[i]
        print('{:10.0f} | {:10.1e} | {:10.1e} | {:10.1e}'.format(i, Fx, Fy, Fz))


    print("\nIf those two tables are the same, it works!")

if __name__ == '__main__':
    test()
