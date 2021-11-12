import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D
from src import forces
from tqdm import tqdm
import datetime
import os
import random
import sys
from time import sleep

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
    #endingKEs = -calculatePEs(masses, endingPositions, endingVelocities) + calculatePEs(masses, startingPositions, startingVelocities) + calculateKEs(masses, startingPositions, startingVelocities)
    #endingVmags = (2*endingKEs/np.array(masses))**0.5
    #endingVelocities = forces.rescale(endingVelocities, endingVmags)

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
                tot[i] -= 6.67e-11*masses[j]/forces.magnitude(positions[i] - positions[j])
    return tot

def animate(masses, positions, velocities, duration, dt, name):    
    # make sure the three input arrays have consistent shapes
    nParticles, nDimensions = positions.shape
    assert(velocities.shape == positions.shape)
    assert(len(masses) == nParticles)

    timesInSecs = np.arange(0, duration, dt)
    timeInDays = timesInSecs / 86400

    wri = ani.FFMpegWriter(fps=60)
    fig = plt.figure(figsize=(10, 10))
    fig.set_facecolor('#000000')
    ax = fig.add_subplot(111, projection='3d')
    ax.w_xaxis.line.set_color('w')
    ax.w_yaxis.line.set_color('w')
    ax.w_zaxis.line.set_color('w')
    ax.w_zaxis.line.set_color('w')
    ax.xaxis.label.set_color('w')
    ax.yaxis.label.set_color('w')
    ax.zaxis.label.set_color('w')
    ax.tick_params(axis='x', colors='w')  # only affects
    ax.tick_params(axis='y', colors='w')  # tick labels
    ax.tick_params(axis='z', colors='w')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_facecolor('#000000')
    # create a filename with system name, date, and time
    filename = './videos/' + name + '_' + \
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.mp4'
    with wri.saving(fig, filename, 200):
        print("Generating video...")
        # calculate appropriate x,y limits for the plot
        x_min_m = np.min(positions[:, 0])
        x_max_m = np.max(positions[:, 0])
        y_min_m = np.min(positions[:, 1])
        y_max_m = np.max(positions[:, 1])
        for i in tqdm(range(len(timeInDays))):
            positions, velocities = updateParticles(masses, positions, velocities, dt)
            x_min_m = min(np.min(positions[:, 0]), x_min_m)
            x_max_m = max(np.max(positions[:, 0]), x_max_m)
            y_min_m = min(np.min(positions[:, 1]), y_min_m)
            y_max_m = max(np.max(positions[:, 1]), y_max_m)
            x_range = x_max_m - x_min_m
            y_range = y_max_m - y_min_m
            x_min = x_min_m - 0.1 * x_range
            x_max = x_max_m + 0.1 * x_range
            y_min = y_min_m - 0.1 * y_range
            y_max = y_max_m + 0.1 * y_range
            # calculate the z limits for the plot as the average of the x and y limits
            z_min = (x_min + y_min) / 2
            z_max = (x_max + y_max) / 2
            ax.clear()
            ax.set_title(f'{name} at {timeInDays[i]:.1f} Days')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            # size the particles by mass (1D array) but no particle is smaller than 1 or bigger than 100
            sizes = np.clip(masses / max(masses) * 200, 5, 200)
            # color the particles by their z velocity
            p = ax.scatter(positions[:, 0], positions[:, 1],
                           positions[:, 2], s=sizes, c=velocities[:, 2], cmap='autumn', alpha=0.8)
            # color bar
            if i == 0:
                cbar = fig.colorbar(p)
                cbar.set_label('z velocity', color='w')
                cbar.ax.yaxis.set_tick_params(color='w')

                # set colorbar edgecolor 
                cbar.outline.set_edgecolor('w')

                # set colorbar ticklabels
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')
            wri.grab_frame()
        # close the writer
        wri.finish()
        sleep(2)
        print("Video generated!")
        # open the video in the default video player
        # os.system(f'open {filename}')
