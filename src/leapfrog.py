import datetime
import os
from time import sleep
import sys

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from src import forces


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def faketype(words, speed=0.001, newline=True):
    for char in words:
        sleep(speed)
        print(char, end="")
        sys.stdout.flush()
    if newline:
        print("")
    return ""


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
    startingForces = np.array(
        forces.calculateForceVectors(masses, startingPositions))

    # calculate the acceleration due to gravity, at the starting position
    startingAccelerations = startingForces / \
        np.array(masses).reshape(nParticles, 1)

    # calculate the ending position
    nudge = startingVelocities*dt + 0.5 * \
        startingAccelerations*dt**2  # nudge = v*dt + 1/2*a*dt^2
    endingPositions = startingPositions + nudge

    # calculate net force vectors on all particles, at the ending position
    endingForces = np.array(
        forces.calculateForceVectors(masses, endingPositions))

    # calculate the acceleration due to gravity, at the ending position
    endingAccelerations = endingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending velocity
    endingVelocities = (startingVelocities +
                        0.5*(endingAccelerations + startingAccelerations)*dt)

    return endingPositions, endingVelocities


def calculateKEs(masses, positions, velocities):
    n = len(masses)
    tot = np.zeros(n)
    for i in range(n):
        tot[i] = 0.5*masses[i]*forces.magnitude(velocities[i])**2
    return tot


def calculatePEs(masses, positions, velocities):
    n = len(masses)
    tot = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if(i != j):
                tot[i] -= 6.67e-11*masses[j] / \
                    forces.magnitude(positions[i] - positions[j])
    return tot


def animate(masses, positions, velocities, duration, dt, name):

    # make sure the three input arrays have consistent shapes
    nParticles, nDimensions = positions.shape
    assert(velocities.shape == positions.shape)
    assert(len(masses) == nParticles)

    timesInSecs = np.arange(0, duration, dt)
    timeInDays = timesInSecs / 86400

    plt.style.use("dark_background")

    # Determine the framerate that results in one year in the simulation taking 15 seconds
    oneyear = 15
    dt_in_days = dt / 86400
    fps_ = round(365 / (oneyear*dt_in_days))

    # Set up the figure
    wri = ani.FFMpegWriter(fps=fps_)
    fig = plt.figure(figsize=(30, 10))
    isometric = fig.add_subplot(132, projection='3d')
    ke_2d = fig.add_subplot(396)
    xz_plane = fig.add_subplot(131)
    xy_plane = fig.add_subplot(133)

    filename = './videos/' + name + '_' + \
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.mp4'

    with wri.saving(fig, filename, 100):
        faketype("Calculating trajectories and generating video...")

        # calculate the distance of each particle to the origin
        distances = np.array([forces.magnitude(positions[i])
                             for i in range(nParticles)])
        normed_distances = distances / np.max(distances)

        # calculate the KE of each particle
        KEs = np.empty((nParticles, len(timesInSecs)))

        cmap = plt.get_cmap('plasma')
        new_cmap = truncate_colormap(cmap, 0.3, 1.0)

        x_min_m = np.min(positions[:, 0])
        x_max_m = np.max(positions[:, 0])
        y_min_m = np.min(positions[:, 1])
        y_max_m = np.max(positions[:, 1])
        z_min_m = np.min(positions[:, 2])
        z_max_m = np.max(positions[:, 2])

        for i in tqdm(range(len(timeInDays))):
            positions, velocities = updateParticles(
                masses, positions, velocities, dt)

            distances = np.array([forces.magnitude(positions[k])
                                  for k in range(nParticles)])
            normed_distances = distances / np.max(distances)

            KEs[:, i] = calculateKEs(
                masses, positions, velocities)

            # this whole business keeps the axes on the same scale and (0,0,0) in the center
            x_min_m = min(np.min(positions[:, 0]), x_min_m)
            x_max_m = max(np.max(positions[:, 0]), x_max_m)
            y_min_m = min(np.min(positions[:, 1]), y_min_m)
            y_max_m = max(np.max(positions[:, 1]), y_max_m)
            z_max_m = max(np.max(positions[:, 2]), z_max_m)
            z_min_m = min(np.min(positions[:, 2]), z_min_m)
            x_range = x_max_m - x_min_m
            y_range = y_max_m - y_min_m
            z_range = z_max_m - z_min_m
            x_min = x_min_m - 0.1 * x_range
            x_max = x_max_m + 0.1 * x_range
            y_min = y_min_m - 0.1 * y_range
            y_max = y_max_m + 0.1 * y_range
            z_min = z_min_m - 0.1 * z_range
            z_max = z_max_m + 0.1 * z_range
            bound = np.abs(max([x_min, x_max, y_min, y_max, z_min, z_max]))

            isometric.clear()
            isometric.set_xlabel("x")
            isometric.set_ylabel("y")
            isometric.set_zlabel("z")
            isometric.set_xlim(-bound, bound)
            isometric.set_ylim(-bound, bound)
            isometric.set_zlim(-bound, bound)
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
            isometric.scatter(positions[:, 0], positions[:, 1],
                              positions[:, 2], s=sizes, c=normed_distances, cmap=new_cmap)  # , alpha=0.8) # the 3d plotting uses variable alphas to show depth

            xz_plane.clear()
            xz_plane.set_xlabel("x")
            xz_plane.set_ylabel("z")
            xz_plane.set_xlim(-bound, bound)
            xz_plane.set_ylim(-bound, bound)
            xz_plane.grid(False)
            xz_plane.facecolor = 'black'
            xz_plane.set_xticks([])
            xz_plane.set_yticks([])
            xz_plane.scatter(positions[:, 0], positions[:, 2],
                             s=sizes, c=normed_distances, cmap=new_cmap, alpha=0.8)

            xy_plane.clear()
            xy_plane.set_xlabel("x")
            xy_plane.set_ylabel("y")
            xy_plane.set_xlim(-bound, bound)
            xy_plane.set_ylim(-bound, bound)
            xy_plane.grid(False)
            xy_plane.facecolor = 'black'
            xy_plane.set_xticks([])
            xy_plane.set_yticks([])
            xy_plane.scatter(positions[:, 0], positions[:, 1],
                             s=sizes, c=normed_distances, cmap=new_cmap, alpha=0.8)  # TODO why does this not show the same color as the other plots?

            ke_2d.clear()
            for j in range(KEs.shape[0]):
                ke_2d.plot(timeInDays[:i], KEs[j, :i],
                           c=cmap(normed_distances[j]))
                ke_2d.set_xlim(timeInDays[0], timeInDays[-1])
                ke_2d.set_xlabel("Time")
                ke_2d.set_ylabel("Kinetic Energy")
                ke_2d.set_xticks([])
                ke_2d.set_yticks([])
            fig.suptitle(f'{name} at {timeInDays[i]:.1f} Days')
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
