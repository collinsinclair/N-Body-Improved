try:
    import datetime
    import os
    import sys
    from time import sleep

    import matplotlib.animation as ani
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from tqdm import tqdm

    from src import leapfrog, systems

    # TODO also need to check that ffmpeg is installed - how to do this from .py script?
except ImportError:
    print("""
    An error occurred while importing one of the required modules.
    Please install required modules with the following command:
    pip3 install -r requirements.txt
    """)
    sys.exit(1)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def makeVideosDir():
    if not os.path.exists("videos"):
        os.mkdir("videos")


def faketype(words, speed=0.001, newline=True):
    for char in words:
        sleep(speed)
        print(char, end="")
        sys.stdout.flush()
    if newline:
        print("")
    return ""


def faketypeIntro():
    # clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    # roll the text onto the screen as if it's being typed by someone
    input(faketype(
        "Welcome to Collin's N-Body Gravity Simulation! Press return to continue.", newline=False))
    input(faketype(
        "This program simulates the time evolution of various gravitational systems like the Sun and Moon, a planetesimal disk, or a star cluster. [return]", newline=False))
    input(faketype(
        "The simulation uses a Runge-Katta Method, a iterative numerical method which we apply to the equations of motion of a system of particles. [return]", newline=False))
    input(faketype("""Each system has parameters that you can change to modify the simulation, which include
    - the total duration (how long the simulation runs)
    - the time step (the simulation time between calculations - smaller = smoother and more accurate, but takes longer)
    - the number of bodies in the system
    - the scatter of starting velocities in the system
    - and more! [return]""", newline=False))
    # get current working directory
    cwd = os.getcwd()
    input(faketype("Each time you run a simulation, the program will store the resulting video in a 'videos' folder that was created in the same folder you ran this program in: {}. [return]".format(
        cwd), newline=False))


def faketypeSystemMenu():
    faketype("--------------------")
    faketype("1. Sun-Earth System", 0.001)
    faketype("2. Sun-Earth-Moon System", 0.001)
    faketype("3. Kepler-16A Circumbinary Planet System", 0.001)
    faketype("4. Random Cube of Bodies", 0.001)
    faketype(
        """5. Uniform Cube of Bodies
    - the positions of the particles start
    on a perfectly uniform grid, but they have
    some initial velocities)""", 0.001)
    faketype("6. Pythagorean System (3-4-5 Triangle)", 0.001)
    faketype(
        """7. Figure 8
    - 3-body initial conditions a classic
    example of N-body choreography, the obscure
    art of finding perfectly periodic N-body solutions""", 0.001)
    faketype(
        """8. Planetesimal Disk
    - initial conditions for a (very) cartoon
    model of a disk of planetesimals (baby planets)
    orbiting around the star""", 0.001)
    faketype(
        """9. Tiny Cluster
    - initial conditions for a (very) cartoon model
    of stellar cluster""", 0.001)
    faketype("--------------------")


def faketypeOptions(chooseN=False, defaultN=0):
    faketype("""
DURATION, TIMESTEP, SAMPLINGRATE, and SPEED
The duration specifies the amount of time the simulation runs. The timestep specifies the amount of time over which to \"nudge\" the particles in each step of the numerical integration. A smaller time step leads to a longer calculation time but gives more accurate results. The sampling rate determines how many steps are skipped between video frames. The speed determines how fast the video plays. 1x speed corresponds to 15 seconds of video per year in simulation.""")
    choice = input("""Would you like to
    (1) use the defaults (1 year, 0.5 days, 1 step/frame, 1x speed) or
    (2) enter your own?
Enter a 1 or 2: """)
    while choice != "1" and choice != "2":
        faketype("Invalid selection. Enter 1 or 2: ", newline=False)
        choice = input()
    if choice == "1":
        duration = 365 * 24 * 60 * 60  # 1 year
        timestep = 60 * 60 * 24 * 0.5  # 0.5 days
        samplingrate = 1
        speed = 1
    else:
        duration = float(input("Enter a duration in years: ")
                         ) * (365 * 60 * 60 * 24)
        timestep = float(input("Enter a time step in days: ")) * (60 * 60 * 24)
        samplingrate = int(
            input("Enter an integer number of steps per frame: "))
        speed = int(input("Enter a speed multiplier: "))
    if chooseN:
        faketype(f"""
NUMBER OF PARTICLES""")
        nChoice = input(f"""Would you like to
    (1) use the default number of particles ({defaultN}) or
    (2) enter your own?
Enter a 1 or a 2: """)
        while nChoice != "1" and nChoice != "2":
            nChoice = input("Invalid selection. Enter a 1 or as 2: ")
        if nChoice == "1":
            n = 0
        else:
            n = int(input("Enter the number of particles: "))
    else:
        n = 0
    return duration, timestep, samplingrate, speed, n


def faketypeVelScatter(default=0):
    faketype(f"""
VELOCITY SCATTER
The velocity scatter specifies the width of the Gaussian distribution from which to draw the initial velocities of the particles. The higher the value, the more spread out the velocities will be.""")
    choice = input(f"""Would you like to
    (1) use the default ({default}) or
    (2) enter your own?
Enter a 1 or a 2: """)
    while choice != "1" and choice != "2":
        choice = input("Invalid selection. Enter a 1 or a 2: ")
    if choice == "2":
        vel_scatter = int(input("Enter the velocity scatter: "))
    else:
        vel_scatter = 0
    return vel_scatter


def faketypeMassRatios(default=0):
    faketype(f"""
MASS RATIO
The evolution of the system depends on the ratio of mass each planetesimal to the mass of the central star. At very small values, the gravity is totally dominated by the central star; at larger values (above about 1e-6, roughly an Earth mass per particle), the orbits may start to go unstable due to the interactions between the particles.""")
    choice = input(f"""Would you like to
    (1) use the default ratio ({default}) or
    (2) enter your own?
Enter a 1 or a 2: """)
    while choice != "1" and choice != "2":
        faketype("Invalid selection")
        choice = input(
            "Would you like to (1) use the default ratio or (2) enter your own? ")
    if choice == "2":
        ratio = float(input("Enter the ratio: "))
    else:
        ratio = 0
    return ratio


def faketypeZVel(default=0):
    faketype(f"""
Z VELOCITY
The z velocity of the particles is the vertical component of their velocity. The z velocity is the component of the velocity in the direction of the z axis.""")
    choice = input(f"""Would you like to
    (1) use the default z velocity ({default}) or
    (2) enter your own?
Enter a 1 or a 2: """)
    while choice != "1" and choice != "2":
        choice = input("Invalid selection. Enter a 1 or a 2: ")
    if choice == "2":
        z_vel = int(input("Enter the z-velocity: "))
    else:
        z_vel = 0
    return z_vel


def faketypeMaxMass(default=0):
    m_sun = 1.98892e30  # kg
    faketype("""
MAXIMUM MASS
The maximum mass of the particles in the cluster is 0.01 solar masses by default.""")
    choice = input(f"""Would you like to
    (1) use the default ({default/m_sun:.2f} solar masses) or
    (2) enter your own?
Enter a 1 or a 2: """)
    while choice != "1" and choice != "2":
        faketype("Invalid selection")
        choice = input(
            "Would you like to (1) use the default or (2) enter your own? ")
    if choice == "2":
        max_mass = m_sun * \
            float(input("Enter the maximum mass (in solar masses): "))
    else:
        max_mass = 0
    return max_mass


def simulateSunEarth():
    masses, positions, velocities = systems.SunEarth()
    duration, dt, samplingrate, speed, n = faketypeOptions()
    return masses, positions, velocities, duration, dt, samplingrate, speed


def simulateSunEarthMoon():
    masses, positions, velocities = systems.SunEarthMoon()
    duration, dt, samplingrate, speed, n = faketypeOptions()
    return masses, positions, velocities, duration, dt, samplingrate, speed


def simulateKepler16():
    masses, positions, velocities = systems.Kepler16()
    duration, dt, samplingrate, speed, n = faketypeOptions()
    return masses, positions, velocities, duration, dt, samplingrate, speed


def simulateRandomCube():
    duration, dt, samplingrate, speed, n = faketypeOptions(
        chooseN=True, defaultN=30)
    if n == 0:
        n = 30
    velScatter = faketypeVelScatter(2000)
    if velScatter == 0:
        velScatter = 2000
    masses, positions, velocities = systems.randomCube(n, velScatter)
    return masses, positions, velocities, duration, dt, samplingrate, speed


def simulateUniformCube():
    duration, dt, samplingrate, speed, n = faketypeOptions(True, 16)
    if n == 0:
        n = 16
    velScatter = faketypeVelScatter(5000)
    if velScatter == 0:
        velScatter = 5000
    masses, positions, velocities = systems.uniformCube(n, velScatter)
    return masses, positions, velocities, duration, dt, samplingrate, speed


def simulatePythagorean():
    masses, positions, velocities = systems.pythagorean()
    duration, dt, samplingrate, speed, n = faketypeOptions()
    return masses, positions, velocities, duration, dt, samplingrate, speed


def simulateFigure8():
    masses, positions, velocities = systems.figure8()
    duration, dt, samplingrate, speed, n = faketypeOptions()
    return masses, positions, velocities, duration, dt, samplingrate, speed


def simulatePlanetesimalDisk():
    duration, dt, samplingrate, speed, n = faketypeOptions(True, 30)
    if n == 0:
        n = 30
    massRatio = faketypeMassRatios(1e-10)
    if massRatio == 0:
        massRatio = 1e-10
    zVel = faketypeZVel(1000)
    if zVel == 0:
        zVel = 1000
    masses, positions, velocities = systems.planetesimalDisk(
        n, massRatio, zVel)
    return masses, positions, velocities, duration, dt, samplingrate, speed


def simulateTinyCluster():
    duration, dt, samplingrate, speed, n = faketypeOptions(True, 20)
    if n == 0:
        n = 20
    maxMass = faketypeMaxMass(0.01*1.989e30)
    if maxMass == 0:
        maxMass = 0.01 * 1.989e30
    masses, positions, velocities = systems.tinyCluster(n, maxMass)
    return masses, positions, velocities, duration, dt, samplingrate, speed


def selectSystem():
    systems = ["SunEarth", "SunEarthMoon", "Kepler16", "RandomCube",
               "UniformCube", "Pythagorean", "Figure8", "PlanetesimalDisk", "TinyCluster"]
    faketypeSystemMenu()
    system = input("Please select a system: ")
    # attempt to convert the input to an integer
    validChoice = False
    while not validChoice:
        try:
            system = int(system)
            if system < 0 or system > 9:
                system = input(
                    "Invalid choice. Please enter a number between 1 and 9: ")
            else:
                validChoice = True
        except ValueError:
            system = input(
                "Invalid choice. Please enter a number between 1 and 9: ")
    if system == 1:
        # clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Sun-Earth")
        return simulateSunEarth(), systems[system-1]
    elif system == 2:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Sun-Earth-Moon")
        return simulateSunEarthMoon(), systems[system-1]
    elif system == 3:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Kepler-16")
        return simulateKepler16(), systems[system-1]
    elif system == 4:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Random Cube")
        return simulateRandomCube(), systems[system-1]
    elif system == 5:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Uniform Cube")
        return simulateUniformCube(), systems[system-1]
    elif system == 6:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Pythagorean")
        return simulatePythagorean(), systems[system-1]
    elif system == 7:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Figure 8")
        return simulateFigure8(), systems[system-1]
    elif system == 8:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Planetesimal Disk")
        return simulatePlanetesimalDisk(), systems[system-1]
    elif system == 9:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Tiny Cluster")
        return simulateTinyCluster(), systems[system-1]


def main():
    makeVideosDir()
    faketypeIntro()
    cont = True
    while cont:
        # clear the screen
        os.system('cls' if os.name == 'nt' else 'clear')
        (masses, positions, velocities, duration, dt,
         samplingrate, speed), name = selectSystem()
        faketype("--------------------")
        leapfrog.animate(masses, positions, velocities,
                         duration, dt, samplingrate, speed, name)
        cont = input("Would you like to run another simulation? (y/n) ")
        while cont != "y" and cont != "n":
            cont = input("Would you like to run another simulation? (y/n) ")
        if cont == "n":
            # clear the screen
            os.system('cls' if os.name == 'nt' else 'clear')
            # get path to video directory
            videoDir = os.path.join(os.getcwd(), "videos")
            # get size of video directroy
            size = 0
            for path, dirs, files in os.walk(videoDir):
                for f in files:
                    fp = os.path.join(path, f)
                    size += os.path.getsize(fp)
            faketype(
                f"Your videos are stored in {videoDir}, and they take up {size/1e6:.1f} MB.")
            faketype("Thank you for using the simulation!")
            cont = False


if __name__ == "__main__":
    main()
