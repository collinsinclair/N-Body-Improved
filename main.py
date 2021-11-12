import datetime
import os
import random
import sys
from time import sleep

import matplotlib.animation as ani
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from src import forces, leapfrog, systems


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
    choice = input(f"""
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
    (2) enter your own?""")
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
    timeInDays = timesInSecs / 86400
    oneyear = 15  # how long one year in simulation should take in real seconds
    dt = (timesInSecs[1] - timesInSecs[0]) / (24 * 3600)
    fps_ = round(365 / (oneyear*dt))
    wri = ani.FFMpegWriter(fps=fps_)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # create an axes below the 3d project for the velocities of the particles
    ax2 = fig.add_subplot(333)
    # create a filename with system name, date, and time
    filename = './videos/' + systemName + '_' + \
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.mp4'
    with wri.saving(fig, filename, 100):
        faketype("Generating video...")
        # calculate appropriate x,y limits for the plot
        # x_min = np.min(positions[:, 0, :])
        # x_max = np.max(positions[:, 0, :])
        # y_min = np.min(positions[:, 1, :])
        # y_max = np.max(positions[:, 1, :])
        # x_range = x_max - x_min
        # y_range = y_max - y_min
        # x_min -= 0.1 * x_range
        # x_max += 0.1 * x_range
        # y_min -= 0.1 * y_range
        # y_max += 0.1 * y_range
        ub = np.quantile(positions[:, 0:2, :], 0.9)
        lb = np.quantile(positions[:, 0:2, :], 0.1)
        # calculate the z limits for the plot as the average of the x and y limits
        # z_min = (x_min + y_min) / 2
        # z_max = (x_max + y_max) / 2
        distances = np.sqrt(
            positions[:, 0, :]**2 + positions[:, 1, :]**2 + positions[:, 2, :]**2)
        # normalize the distances to the range [0, 1]
        normed_distances = distances / np.max(distances)
        # calculate absolute velocities
        absVelocities = np.sqrt(
            velocities[:, 0, :]**2 + velocities[:, 1, :]**2 + velocities[:, 2, :]**2)
        # get plasma colormap
        cmap = plt.get_cmap('plasma')
        new_cmap = truncate_colormap(cmap, 0.3, 1.0)
        # calculate the kinetic energy of each particle with the same dimensions as absVelocities
        kineticEnergy = 0.5 * masses[:, None] * absVelocities**2
        normed_KE = kineticEnergy / np.max(kineticEnergy)
        scaled_KE = kineticEnergy / np.min(kineticEnergy)
        # calculate the position of the center of gravity
        # x component
        x_cg = np.sum(masses[:, None, None] *
                      positions[:, 0, 0]) / np.sum(masses)
        # y component
        y_cg = np.sum(masses[:, None, None] *
                      positions[:, 1, 0]) / np.sum(masses)
        # z component
        z_cg = np.sum(masses[:, None, None] *
                      positions[:, 2, 0]) / np.sum(masses)
        cg = np.array([x_cg, y_cg, z_cg])
        # calculate the distance of each particle from the center of gravity
        # distancesFromCOG = np.sqrt((positions[:, 0, :] - cg[0])**2 +
        #                            (positions[:, 1, :] - cg[1])**2 +
        #                            (positions[:, 2, :] - cg[2])**2)
        # normedDistancesFromCOG = distancesFromCOG / np.max(distancesFromCOG)
        for i in tqdm(range(len(timeInDays))):
            ax.clear()
            ax.set_title(f'{systemName} at {timeInDays[i]:.1f} Days')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(lb, ub)
            ax.set_ylim(lb, ub)
            ax.set_zlim(lb, ub)
            # Hide grid lines
            ax.grid(False)
            ax.facecolor = 'black'
            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('k')
            ax.yaxis.pane.set_edgecolor('k')
            ax.zaxis.pane.set_edgecolor('k')
            sizes = np.clip(masses / max(masses) * 300, 10, 300)
            # give each particle a different color based on its distance from the origin
            sp = ax.scatter(positions[:, 0, i], positions[:, 1, i],
                            positions[:, 2, i], s=sizes, c=normed_distances[:, i], cmap=new_cmap)
            ax2.clear()
            # line plot of each particle's velocity as a function of time colored by mass
            for j in range(len(absVelocities)):
                p = ax2.plot(timeInDays[:i], kineticEnergy[j, :i],
                             c=new_cmap(normed_distances[j, i]))
                ax2.set_xlim(timeInDays[0], timeInDays[-1])
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Kinetic Energy")
                ax2.set_xticks([])
                ax2.set_yticks([])
            # if i == 0:
            #     cb = fig.colorbar(sp, ax=ax2)
            #     cb.set_label("Distance from Barycenter")
            wri.grab_frame()
        # close the writer
        wri.finish()
        print("Finishing up...")
        sleep(5)
        faketype("Video generated!")
        # attempt to open the video in the default video player
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
