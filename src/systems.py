import numpy as np

G = 6.67408e-11
au = 1.496e+11
sun_mass = 1.989e30
pc = 3.086e+16
day = 24.0*60.0*60.0


def sun_earth():
    """This function creates initial conditions for the Earth, Sun system."""

    # mass of the Sun, mass of the Earth, in kg
    masses = np.array([1.989e30, 5.972e24])

    # center of mass separation
    a = 1.0*au
    # noinspection PyPep8Naming
    M = np.sum(masses)
    q = masses[1]/M

    v_circular = np.sqrt(G*M/a)

    positions = np.array([[-q*a, 0.0, 0.0], [(1-q)*a, 0.0, 0.0]])
    velocities = np.array([[0.0, -q*v_circular, 0.0],
                           [0.0, (1-q)*v_circular, 0.0]])

    return masses, positions, velocities


def kepler_16():
    """This function creates initial conditions for
    the Kepler-16A circumbinary planet system."""

    masses = np.array([0.6897, 0.20255]) * sun_mass

    mu = masses[0]*masses[1]/np.sum(masses)

    # q = masses[1]/np.sum(masses)
    r = np.array([33632559799.154907, -4254739.055706108, 8125952284.81131])
    v = np.array([-4626.159061662148, -30.053284017468425, 57397.53924516573])

    r_a = mu/masses[0]*r
    v_a = mu/masses[0]*v

    r_b = -mu/masses[1]*r
    v_b = -mu/masses[1]*v

    # vA = -q*vB
    # rA = -q*rB
    r = np.array([r_a, r_b])
    v = np.array([v_a, v_b])
    # rcm, vcm = calculateCenterOfMass(masses, r, v)
    # r -= rcm
    # v -= vcm

    r_c = np.array([37761268405.67878,
                    -40923370.95730546,
                    81934000554.43219])
    v_c = np.array([-36297.78857577688,
                    -10.812291771331552,
                    17020.175739138824])

    masses = np.hstack([masses, 0.000318 * sun_mass])
    positions = np.vstack([r, r_c])
    velocities = np.vstack([v, v_c])

    return masses, positions, velocities


def sun_earth_moon():
    """This function creates initial conditions for the Earth, Moon,
    Sun system. """

    # masses, positions, velocities drawn from JPL HORIZONS
    masses = np.array([1.988544e30, 5.97219e24, 734.9e20])

    positions = np.array([[5.258735436560317E+05, 5.345075371093444E+05,
                           -2.375056567675160E+04],
                          [1.038830938047841E+08, 1.067597535723964E+08,
                           -2.841144231139123E+04],
                          [1.041686592750002E+08, 1.065011138891447E+08,
                           -1.549483993739635E+04]])*1e3

    velocities = np.array([[-3.799443100144098E-03, 1.173726161867020E-02,
                            7.838374539311920E-05],
                           [-2.184611793623773E+01, 2.066569272798333E+01,
                            4.404945193954291E-04],
                           [-2.121635210223251E+01, 2.145773100146389E+01,
                            -8.484959015984650E-02]])*1e3

    return masses, positions, velocities


def random_cube(n=30, velocity_scatter=2000.0, seed=None):
    """This function creates N-body initial conditions for a cube .

    Inputs:
        N (= 30 by default)
            the total number of particles to create,
            with the first particle being a solar-mass star
        velocity_scatter (= 2000.0 by default)
            particles will be given velocities drawn from a
            Gaussian distribution with a width of velocity_scatter
        seed (= None by default)
            a seed for the random number generator, any integer

    Outputs:
        this function returns three arrays
            masses [shape is (nParticles)]
            positions [shape (nParticles, nDimensions)]
            velocities [shape (nParticles, nDimensions)]

    Example Usage:
        mParticles, initialPositions, initialVelocities = randomCube()
    """

    # set the seed
    np.random.seed(seed)

    # uniformly distributed random masses below 1 solar mass
    masses = np.random.uniform(0, 1, n) * sun_mass * 0.1

    # random cube of positions
    positions = np.random.uniform(-1, 1, [n, 3])*au

    # randomized velocities
    velocities = velocity_scatter*np.random.normal(0, 1, [n, 3])

    return masses, positions, velocities


def uniform_cube(n=16, velocity_scatter=5000.0, seed=None):
    """This function creates N-body initial conditions for a cube of
    particles, where the positions of the particles start on a perfectly
    uniform grid, but they have some initial velocities.

    Inputs:
        N (= 16 by default)
            the total number of particles to create,
            with the first particle being a solar-mass star
        velocity_scatter (= 5000.0 by default)
            particles will be given velocities drawn from a
            Gaussian distribution with a width of velocity_scatter
        seed (= None by default)
            a seed for the random number generator, any integer

    Outputs:
        this function returns three arrays
            masses [shape is (nParticles)]
            positions [shape (nParticles, nDimensions)]
            velocities [shape (nParticles, nDimensions)]

    Example Usage:
        mParticles, initialPositions, initialVelocities = uniformCube()
    """

    # set the seed
    np.random.seed(seed)

    # how many particles per side of the cube should we use?
    n_per_side = np.ceil(n**(1.0/3.0)).astype(np.int)

    # constant masses
    masses = np.ones(n_per_side**3) * sun_mass * 0.01

    side = np.linspace(-0.5, 0.5, n_per_side)*au
    x, y, z = np.meshgrid(side, side, side)
    positions = np.array([x.flatten(), y.flatten(), z.flatten()]).T

    # randomized velocities
    velocities = velocity_scatter*np.random.normal(0, 1, [n_per_side**3, 3])

    return masses, positions, velocities


def pythagorean():
    """This function creates initial conditions for
    a 3-4-5 right triangle, with equal masses. There are
    some very close approaches that would occur in here, so
    see the note below in "tinyCluster" regarding how to
    handle this issue.

    Inputs:
        (none)

    Outputs:
        this function returns three arrays
            masses [shape is (nParticles)]
            positions [shape (nParticles, nDimensions)]
            velocities [shape (nParticles, nDimensions)]

    Example Usage:
        mParticles, initialPositions, initialVelocities = pythagorean()
    """

    # three equal mass objects
    masses = np.ones(3) * sun_mass

    # the corners of a 3, 4, 5 triangle
    positions = np.array([[0.0, 0.0, 0.0],
                          [4.0, 0.0, 0.0],
                          [0.0, 3.0, 0.0]])*au
    positions -= np.mean(positions, 0)

    # start them initially from rest
    velocities = np.zeros_like(positions)

    return masses, positions, velocities


def figure8():
    """This function creates 3-body initial conditions a classic
    example of N-body choreography, the obscure art of finding
    perfectly periodic N-body solutions.

    Inputs:
        (none)

    Outputs:
        this function returns three arrays
            masses [shape is (nParticles)]
            positions [shape (nParticles, nDimensions)]
            velocities [shape (nParticles, nDimensions)]

    Example Usage:
        mParticles, initialPositions, initialVelocities = figure8()
    """

    masses = np.ones(3) * sun_mass
    # finish this!
    period = 200*24*60*60
    a = (G * sun_mass * period ** 2 / 4 / np.pi ** 2) ** (1.0 / 3.0)
    v = np.sqrt(G * sun_mass / a)
    x2 = 0.995492
    x1 = -x2
    x3 = 0.0
    y1 = 0
    y2 = 0
    y3 = 0

    vx3 = 0.695804
    vy3 = 1.067860
    vy2 = -vy3/2
    vy1 = -vy3/2
    vx1 = -vx3/2
    vx2 = -vx3/2
    positions = np.array([[x1, y1, 0], [x2, y2, 0], [x3, y3, 0]])*a
    velocities = np.array([[vx1, vy1, 0], [vx2, vy2, 0], [vx3, vy3, 0]])*v

    return masses, positions, velocities


def planetesimal_disk(n=30, mass_ratios=1e-10, z_velocity=1000.0):
    """This function creates N-body initial conditions for a (very) cartoon
    model of a disk of planetesimals (baby planets) orbiting around the star.

    Inputs:
        N (= 30 by default)
            the total number of particles to create,
            with the first particle being a solar-mass star
        mass_ratios (= 1e-10 by default)
            the ratio of the mass of each planetesimal to
            the mass of the central star. At very small values,
            the gravity is totally dominated by the central star;
            at larger values (above about 1e-6, roughly an Earth
            mass per particle), the orbits may start to go unstable
            due to the interactions between the particles.
        z_velocity (= 1000.0 by default)
            the disk will be created mostly in the x-y plane,
            but you can add random velocities with a standard
            deviation set by z_velocity (in units of m/s) to
            perturb their orbits above and below the plane

    Outputs:
        this function returns three arrays
            masses [shape is (nParticles)]
            positions [shape (nParticles, nDimensions)]
            velocities [shape (nParticles, nDimensions)]

    Example Usage:
        mParticles, initialPositions, initialVelocities = planetesimalDisk()
    """

    # set up the masses
    masses = np.ones(n) * sun_mass
    masses[1:] *= mass_ratios

    # random positions in the disk
    radii = np.random.uniform(0.3, 1.0, n)*au
    theta = np.random.uniform(0, 2*np.pi, n)
    radii[0] = 0

    # convert to cartesian coordinates
    x = radii*np.cos(theta)
    y = radii*np.sin(theta)
    z = np.random.normal(0, 1, n)*0
    positions = np.vstack([x, y, z]).T

    # calculate velocities for circular orbits
    s = np.zeros(n)
    s[1:] = np.sqrt((G * sun_mass) / radii[1:])
    vx = -np.sin(theta)*s
    vy = np.cos(theta)*s
    vz = np.random.normal(0, 1, n)*z_velocity
    vz[0] = 0
    velocities = np.vstack([vx, vy, vz]).T

    # return them as three separate arrays
    return masses, positions, velocities


# noinspection SpellCheckingInspection
def tiny_cluster(n=20, maximum_mass=0.01 * sun_mass):
    """This function creates N-body initial conditions for
    a (very) cartoon model of stellar cluster.

    WARNING: With these initial conditions, it's very easy
    for some of your particles to have very close approaches.
    This means, to properly resolve their motion, you either
    need to:

        (a) take very short time steps so you accurately
        capture the accelerations of these close approaches

        (b) modify your force of gravity calculation by
        including a "softening length". That is, in the
        place where you calculate F = GMm/r**2, you instead
        calculate the magnitude of the force as GMm/s**(2,
        where s = np.sqrt(r**2 + epsilon**2) where epsilon
        is some small number like 0.1 AU. This "softens" the
        strong forces that would otherwise result from very
        close approaches.

    Inputs:
        N (= 30 by default)
            the total number of particles to create
        maximum_mass (= 0.01 solar masses by default)
            the maximum mass of the particles that can go
            into the cluster; the masses of the particles
            will be random, drawn from a uniform distribution
            with this as the maximum

    Outputs:
        this function returns three arrays
            masses [shape is (nParticles)]
            positions [shape (nParticles, nDimensions)]
            velocities [shape (nParticles, nDimensions)]

    Example Usage:
        mParticles, initialPositions, initialVelocities = tinyCluster()
    """

    # set up the masses
    masses = np.random.uniform(0, 1, n)*maximum_mass

    # convert to cartesian coordinates
    positions = np.random.normal(0, 1.0, [n, 3])*au
    radii = np.sqrt(np.sum(positions**2, 1))
    mass_enclosed = np.array([np.sum(masses[radii <= r]) for r in radii])
    sigma = np.sqrt(G*mass_enclosed/radii)

    # directions = np.array([np.cross(np.random.uniform(size=3), positions[
    # i]) for i in range(N)]) for i in range(N): directions[i,
    # :] /= np.sqrt(np.sum(directions[i,:]**2))

    # calculate velocities for circular orbits
    # velocities = (sigma*np.random.normal(0,1,N))[:,np.newaxis]*directions
    velocities = sigma[:, np.newaxis]*np.random.normal(0, 1, [n, 3])*0.5

    # return them as three separate arrays
    return masses, positions, velocities
