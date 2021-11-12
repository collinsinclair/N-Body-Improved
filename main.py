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
    input(faketype("This program simulates the time evolution of various gravitational systems like the Sun and Moon, a planetesimal disk, or a star cluster.", newline=False))
    input(faketype("The simulation is based on the Leapfrog method, which is a numerical method for solving the equations of motion of a system of particles.", newline=False))
    input(faketype("""Each system has parameters that you can change to modify the simulation, which include
    - the total duration (how long the simulation runs)
    - the time step (the simulation time between calculations - smaller = smoother and more accurate, but takes longer)
    - the number of bodies in the system
    - the scatter of starting velocities in the system
    - and more!""", newline=False))
    # get current working directory
    cwd = os.getcwd()
    input(faketype("Each time you run a simulation, the program will store the resulting video in a 'videos' folder that was created in the same folder you ran this program in: {}.".format(cwd), newline=False))


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
DURATION and TIMESTEP
The duration specifies the amount of time the simulation runs. The timestep specifies the amount of time over which to \"nudge\" the particles in each step of the numerical integration. A smaller time step leads to a longer calculation time but gives more accurate results.""")
    choice = input("""Would you like to
    (1) use the defaults (1 year, 0.5 days) or
    (2) enter your own?
Enter a 1 or 2: """)
    while choice != "1" and choice != "2":
        faketype("Invalid selection. Enter 1 or 2: ", newline=False)
        choice = input()
    if choice == "1":
        duration = 365 * 24 * 60 * 60  # 1 year
        timestep = 60 * 60 * 24 * 0.5  # 0.5 days
    else:
        duration = float(input("Enter a duration in years: ")
                         ) * (365 * 60 * 60 * 24)
        timestep = float(input("Enter a time step in days: ")) * (60 * 60 * 24)
    if chooseN:
        nChoice = input(f"""
NUMBER OF PARTICLES
Would you like to
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
    return duration, timestep, n


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
    faketype("""
MAXIMUM MASS
The maximum mass of the particles in the cluster is 0.01 solar masses by default.""")
    choice = input(f"""Would you like to
    (1) use the default ({default}) or
    (2) enter your own?
Enter a 1 or a 2: """)
    while choice != "1" and choice != "2":
        faketype("Invalid selection")
        choice = input(
            "Would you like to (1) use the default or (2) enter your own? ")
    if choice == "2":
        max_mass = float(input("Enter the maximum mass: "))
    else:
        max_mass = 0
    return max_mass


def simulateSunEarth():
    masses, positions, velocities = systems.SunEarth()
    duration, dt, n = faketypeOptions()
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def simulateSunEarthMoon():
    masses, positions, velocities = systems.SunEarthMoon()
    duration, dt, n = faketypeOptions()
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def simulateKepler16():
    masses, positions, velocities = systems.Kepler16()
    duration, dt, n = faketypeOptions()
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def simulateRandomCube():
    duration, dt, n = faketypeOptions(chooseN=True, defaultN=30)
    if n == 0:
        n = 30
    velScatter = faketypeVelScatter(2000)
    if velScatter == 0:
        velScatter = 2000
    masses, positions, velocities = systems.randomCube(n, velScatter)
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def simulateUniformCube():
    duration, dt, n = faketypeOptions(True, 16)
    if n == 0:
        n = 16
    velScatter = faketypeVelScatter(5000)
    if velScatter == 0:
        velScatter = 5000
    masses, positions, velocities = systems.uniformCube(n, velScatter)
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def simulatePythagorean():
    masses, positions, velocities = systems.pythagorean()
    duration, dt, n = faketypeOptions()
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def simulateFigure8():
    masses, positions, velocities = systems.figure8()
    duration, dt, n = faketypeOptions()
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def simulatePlanetesimalDisk():
    duration, dt, n = faketypeOptions(True, 30)
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
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def simulateTinyCluster():
    duration, dt, n = faketypeOptions(True, 20)
    if n == 0:
        n = 20
    maxMass = faketypeMaxMass(0.01*1.989e30)
    if maxMass == 0:
        maxMass = 0.01 * 1.989e30
    masses, positions, velocities = systems.tinyCluster(n, maxMass)
    times, allPositions, allVelocities = leapfrog.calculateTrajectories(
        masses, positions, velocities, duration, dt)
    return times, allPositions, allVelocities, masses


def animateTrajectories(timesInSecs, positions, velocities, masses, systemName):
    plt.style.use("dark_background")

    # Determine the framerate that results in one year in the simulation taking 15 seconds
    timeInDays = timesInSecs / 86400
    oneyear = 15
    dt = (timesInSecs[1] - timesInSecs[0]) / (24 * 3600)
    fps_ = round(365 / (oneyear*dt))

    # Set up the figure
    wri = ani.FFMpegWriter(fps=fps_)
    fig = plt.figure(figsize=(30, 10))
    isometric = fig.add_subplot(132, projection='3d')
    ke_2d = fig.add_subplot(396)
    xz_plane = fig.add_subplot(131)
    xy_plane = fig.add_subplot(133)

    filename = './videos/' + systemName + '_' + \
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.mp4'

    with wri.saving(fig, filename, 100):
        faketype("Generating video...")
        if len(masses) > 3:
            ub = np.quantile(positions[:, 0:2, :], 0.9)
            lb = np.quantile(positions[:, 0:2, :], 0.1)
        else:
            ub = np.max(positions[:, 0:2, :])
            lb = np.min(positions[:, 0:2, :])
        if ub > lb:
            lb = -ub
        else:
            ub = -lb
        distances = np.sqrt(
            positions[:, 0, :]**2 + positions[:, 1, :]**2 + positions[:, 2, :]**2)
        normed_distances = distances / np.max(distances)
        absVelocities = np.sqrt(
            velocities[:, 0, :]**2 + velocities[:, 1, :]**2 + velocities[:, 2, :]**2)
        cmap = plt.get_cmap('plasma')
        new_cmap = truncate_colormap(cmap, 0.3, 1.0)
        kineticEnergy = 0.5 * masses[:, None] * absVelocities**2
        for i in tqdm(range(len(timeInDays))):
            isometric.clear()
            isometric.set_xlabel("x")
            isometric.set_ylabel("y")
            isometric.set_zlabel("z")
            isometric.set_xlim(lb, ub)
            isometric.set_ylim(lb, ub)
            isometric.set_zlim(lb, ub)
            isometric.grid(False)
            isometric.facecolor = 'black'
            isometric.set_xticks([])
            isometric.set_yticks([])
            isometric.set_zticks([])
            isometric.xaxis.pane.fill = False
            isometric.yaxis.pane.fill = False
            isometric.zaxis.pane.fill = False
            isometric.xaxis.pane.set_edgecolor('k')
            isometric.yaxis.pane.set_edgecolor('k')
            isometric.zaxis.pane.set_edgecolor('k')
            sizes = np.clip(masses / max(masses) * 300, 10, 300)
            isometric.scatter(positions[:, 0, i], positions[:, 1, i],
                              positions[:, 2, i], s=sizes, c=normed_distances[:, i], cmap=new_cmap)

            xtra = 0.5
            xz_plane.clear()
            xz_plane.set_xlabel("x")
            xz_plane.set_ylabel("z")
            xz_plane.set_xlim(lb, ub)
            xz_plane.set_ylim(lb, ub)
            xz_plane.grid(False)
            xz_plane.facecolor = 'black'
            xz_plane.set_xticks([])
            xz_plane.set_yticks([])
            xz_plane.scatter(positions[:, 0, i], positions[:, 2, i],
                             s=sizes, c=normed_distances[:, i], cmap=new_cmap)

            xy_plane.clear()
            xy_plane.set_xlabel("x")
            xy_plane.set_ylabel("y")
            xy_plane.set_xlim(lb, ub)
            xy_plane.set_ylim(lb, ub)
            xy_plane.grid(False)
            xy_plane.facecolor = 'black'
            xy_plane.set_xticks([])
            xy_plane.set_yticks([])
            xy_plane.scatter(positions[:, 0, i], positions[:, 1, i],
                             s=sizes, c=normed_distances[:, i], cmap=new_cmap)

            ke_2d.clear()
            for j in range(len(absVelocities)):
                p = ke_2d.plot(timeInDays[:i], kineticEnergy[j, :i],
                               c=new_cmap(normed_distances[j, i]))
                ke_2d.set_xlim(timeInDays[0], timeInDays[-1])
                ke_2d.set_xlabel("Time")
                ke_2d.set_ylabel("Kinetic Energy")
                ke_2d.set_xticks([])
                ke_2d.set_yticks([])
            fig.suptitle(f'{systemName} at {timeInDays[i]:.1f} Days')
            fig.tight_layout()
            wri.grab_frame()
        wri.finish()
        print("Finishing up...")
        sleep(5)
        faketype("Video generated!")
        try:
            os.system(f'open "{filename}"')
        except:
            faketype("Could not open video in default video player.")


def main():
    makeVideosDir()
    faketypeIntro()
    systems = ["Sun-Earth", "Sun-Earth-Moon", "Kepler16", "Random Cube",
               "Uniform Cube", "Pythagorean", "Figure 8", "Planetesimal Disk", "Tiny Cluster"]
    cont = True
    while cont:
        faketypeSystemMenu()
        system = input("Please select a system: ")
        # attempt to convert the input to an integer
        validChoice = False
        while not validChoice:
            try:
                system = int(system)
                if system < 0 or system > len(systems):
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
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulateSunEarth()
            faketype("--------------------")
        elif system == 2:
            os.system('cls' if os.name == 'nt' else 'clear')
            faketype("--------------------")
            faketype("Sun-Earth-Moon")
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulateSunEarthMoon()
            faketype("--------------------")
        elif system == 3:
            os.system('cls' if os.name == 'nt' else 'clear')
            faketype("--------------------")
            faketype("Kepler-16")
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulateKepler16()
            faketype("--------------------")
        elif system == 4:
            os.system('cls' if os.name == 'nt' else 'clear')
            faketype("--------------------")
            faketype("Random Cube")
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulateRandomCube()
            faketype("--------------------")
        elif system == 5:
            os.system('cls' if os.name == 'nt' else 'clear')
            faketype("--------------------")
            faketype("Uniform Cube")
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulateUniformCube()
            faketype("--------------------")
        elif system == 6:
            os.system('cls' if os.name == 'nt' else 'clear')
            faketype("--------------------")
            faketype("Pythagorean")
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulatePythagorean()
            faketype("--------------------")
        elif system == 7:
            os.system('cls' if os.name == 'nt' else 'clear')
            faketype("--------------------")
            faketype("Figure 8")
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulateFigure8()
            faketype("--------------------")
        elif system == 8:
            os.system('cls' if os.name == 'nt' else 'clear')
            faketype("--------------------")
            faketype("Planetesimal Disk")
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulatePlanetesimalDisk()
            faketype("--------------------")
        elif system == 9:
            os.system('cls' if os.name == 'nt' else 'clear')
            faketype("--------------------")
            faketype("Tiny Cluster")
            faketype("--------------------")
            times, allPositions, allVelocities, masses = simulateTinyCluster()
            faketype("--------------------")
        animateTrajectories(times, allPositions,
                            allVelocities, masses, systems[system - 1])
        cont = input("Would you like to run another simulation? (y/n) ")
        while cont != "y" and cont != "n":
            cont = input("Would you like to run another simulation? (y/n) ")
        if cont == "n":
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
