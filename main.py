import datetime
import os
import random
import sys
from time import sleep

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from src import forces, leapfrog, systems


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


def printIntro():
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


def printSystemMenu():
    print("--------------------")
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
    print("--------------------")


def printOptions(chooseN=False):
    print("The simulation needs a duration and timestep.")
    choice = input(
        "Would you like to (1) use the defaults or (2) enter your own? ")
    while choice != "1" and choice != "2":
        print("Invalid selection")
        choice = input()
    if choice == "1":
        duration = 365 * 24 * 60 * 60  # 1 year
        timestep = 60 * 60 * 24  # 1 day
    else:
        duration = float(input("Enter a duration in years: ")
                         ) * (365 * 60 * 60 * 24)
        timestep = float(input("Enter a time step in days: ")) * (60 * 60 * 24)
    if chooseN:
        nChoice = input(
            "Would you like to (1) use the default number of particles or (2) enter your own? ")
        while nChoice != "1" and nChoice != "2":
            print("Invalid selection")
            nChoice = input(
                "Would you like to (1) use the default number of particles or (2) enter your own? ")
        if nChoice == "1":
            n = 0
        else:
            n = int(input("Enter the number of particles: "))
    else:
        n = 0
    return duration, timestep, n


def printVelScatter():
    choice = input("The particles in this system wil be given velocities drawn from a Gaussian distribution with width X. Would you like to (1) use the default X or (2) enter your own? ")
    while choice != "1" and choice != "2":
        print("Invalid selection")
        choice = input(
            "Would you like to (1) use the default X or (2) enter your own? ")
    if choice == "2":
        vel_scatter = int(input("Enter the velocity scatter: "))
    else:
        vel_scatter = 0
    return vel_scatter


def printMassRatios():
    choice = input("The evolution of the system depends on the ratio of mass each planetesimal to the mass of the central star. At very small values, the gravity is totally dominated by the central star; at larger values (above about 1e-6, roughly an Earth mass per particle), the orbits may start to go unstable due to the interactions between the particles. Would you like to (1) use the default ratio or (2) enter your own? ")
    while choice != "1" and choice != "2":
        print("Invalid selection")
        choice = input(
            "Would you like to (1) use the default ratio or (2) enter your own? ")
    if choice == "2":
        ratio = float(input("Enter the ratio: "))
    else:
        ratio = 0
    return ratio


def printZVel():
    choice = input("The evolution also depends on the z-velocity (the up-down velocity, relative to the flat disk) of the bodies. Would you like to (1) use the default z-velocity or (2) enter your own? ")
    while choice != "1" and choice != "2":
        print("Invalid selection")
        choice = input(
            "Would you like to (1) use the default z-velocity (1000) or (2) enter your own? ")
    if choice == "2":
        z_vel = int(input("Enter the z-velocity: "))
    else:
        z_vel = 0
    return z_vel


def printMaxMass():
    choice = input(
        "The maximum mass of the particles in the cluster is 0.01 solar masses by default. Would you like to (1) use the default or (2) enter your own? ")
    while choice != "1" and choice != "2":
        print("Invalid selection")
        choice = input(
            "Would you like to (1) use the default or (2) enter your own? ")
    if choice == "2":
        max_mass = float(input("Enter the maximum mass: "))
    else:
        max_mass = 0
    return max_mass


def simulateSunEarth():
    masses, positions, velocities = systems.SunEarth()
    duration, dt, n = printOptions()
    return masses, positions, velocities, duration, dt


def simulateSunEarthMoon():
    masses, positions, velocities = systems.SunEarthMoon()
    duration, dt, n = printOptions()
    return masses, positions, velocities, duration, dt


def simulateKepler16():
    masses, positions, velocities = systems.Kepler16()
    duration, dt, n = printOptions()
    return masses, positions, velocities, duration, dt


def simulateRandomCube():
    duration, dt, n = printOptions(True)
    if n == 0:
        n = 30
    velScatter = printVelScatter()
    if velScatter == 0:
        velScatter = 2000
    masses, positions, velocities = systems.randomCube(n, velScatter)
    return masses, positions, velocities, duration, dt


def simulateUniformCube():
    duration, dt, n = printOptions(True)
    if n == 0:
        n = 16
    velScatter = printVelScatter()
    if velScatter == 0:
        velScatter = 5000
    masses, positions, velocities = systems.uniformCube(n, velScatter)
    return masses, positions, velocities, duration, dt


def simulatePythagorean():
    masses, positions, velocities = systems.pythagorean()
    duration, dt, n = printOptions()
    return masses, positions, velocities, duration, dt


def simulateFigure8():
    masses, positions, velocities = systems.figure8()
    duration, dt, n = printOptions()
    return masses, positions, velocities, duration, dt


def simulatePlanetesimalDisk():
    duration, dt, n = printOptions(True)
    if n == 0:
        n = 30
    massRatio = printMassRatios()
    if massRatio == 0:
        massRatio = 1e-10
    zVel = printZVel()
    if zVel == 0:
        zVel = 1000
    masses, positions, velocities = systems.planetesimalDisk(
        n, massRatio, zVel)
    return masses, positions, velocities, duration, dt

def simulateTinyCluster():
    duration, dt, n = printOptions(True)
    if n == 0:
        n = 20
    maxMass = printMaxMass()
    if maxMass == 0:
        maxMass = 0.01 * 1.989e30
    masses, positions, velocities = systems.tinyCluster(n, maxMass)
    return masses, positions, velocities, duration, dt


def animateTrajectories(timesInSecs, positions, velocities, masses, systemName):
    timeInDays = timesInSecs / 86400
    wri = ani.FFMpegWriter(fps=60)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # create a filename with system name, date, and time
    filename = './videos/' + systemName + '_' + \
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.mp4'
    with wri.saving(fig, filename, 200):
        print("Generating video...")
        # calculate appropriate x,y limits for the plot
        x_min = np.min(positions[:, 0, :])
        x_max = np.max(positions[:, 0, :])
        y_min = np.min(positions[:, 1, :])
        y_max = np.max(positions[:, 1, :])
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        # calculate the z limits for the plot as the average of the x and y limits
        z_min = (x_min + y_min) / 2
        z_max = (x_max + y_max) / 2
        for i in tqdm(range(len(timeInDays))):
            ax.clear()
            ax.set_title(f'{systemName} at {timeInDays[i]:.1f} Days')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            # size the particles by mass (1D array) but no particle is smaller than 1 or bigger than 100
            sizes = np.clip(masses / max(masses) * 200, 5, 200)
            # color the particles by their z velocity
            p = ax.scatter(positions[:, 0, i], positions[:, 1, i],
                           positions[:, 2, i], s=sizes, c=velocities[:, 2, i], cmap='viridis')
            # color bar
            if i == 0:
                cbar = fig.colorbar(p)
                cbar.set_label('z velocity')
            wri.grab_frame()
        # close the writer
        wri.finish()
        sleep(2)
        print("Video generated!")
        # open the video in the default video player
        # os.system(f'open {filename}')

def selectSystem():
    systems = ["SunEarth", "SunEarthMoon", "Kepler16", "RandomCube",
               "UniformCube", "Pythagorean", "Figure8", "PlanetesimalDisk", "TinyCluster"]
    printSystemMenu()
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
    printIntro()
    cont = True
    while cont:
        (masses, positions, velocities, duration, dt), name = selectSystem()
        faketype("--------------------")
        leapfrog.animate(masses, positions, velocities, duration, dt, name)
        cont = input("Would you like to run another simulation? (y/n) ")
        while cont != "y" and cont != "n":
            cont = input("Would you like to run another simulation? (y/n) ")
        if cont == "n":
            cont = False


if __name__ == "__main__":
    main()
