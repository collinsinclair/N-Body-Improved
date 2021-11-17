import datetime
import os
import sys
from time import sleep

import matplotlib.animation as ani
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
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


def calculate_acceleration(masses, positions):
    n_particles, n_dimensions = positions.shape
    force = forces.calculateForceVectors(masses, positions)
    acceleration = force / np.array(masses).reshape(n_particles, 1)
    return acceleration


def rungekutta(masses, positions, velocities, dt):
    n_particles, n_dimensions = positions.shape
    kv = np.zeros((4, n_particles, n_dimensions))
    kr = np.zeros((4, n_particles, n_dimensions))
    kv[0] = calculate_acceleration(masses, positions)
    kr[0] = velocities
    kv[1] = calculate_acceleration(masses, positions + kr[0] * dt / 2)
    kr[1] = velocities + kv[0] * dt / 2
    kv[2] = calculate_acceleration(masses, positions + kr[1] * dt / 2)
    kr[2] = velocities + kv[1] * dt / 2
    kv[3] = calculate_acceleration(masses, positions + kr[2] * dt)
    kr[3] = velocities + kv[2] * dt
    nv = dt / 6 * (kv[0] + 2 * kv[1] + 2 * kv[2] + kv[3])
    nr = dt / 6 * (kr[0] + 2 * kr[1] + 2 * kr[2] + kr[3])
    return nr, nv


def update_particles(masses, positions, velocities, dt):
    nr, nv = rungekutta(masses, positions, velocities, dt)
    nr2, nv2, dtm2 = update_particles_recursive(masses, positions, velocities, dt, nr)
    cnr = np.concatenate((np.array([positions]), nr2)).cumsum(axis=0)
    cnv = np.concatenate((np.array([velocities]), nv2)).cumsum(axis=0)
    cdtm = np.concatenate((np.array([0]), dtm2)).cumsum()
    return cnr, cnv, cdtm


def update_particles_recursive(masses, positions, velocities, dt, prev):
    n_particles, n_dimensions = positions.shape
    nr1, nv1 = rungekutta(masses, positions, velocities, dt / 2)
    nr2, nv2 = rungekutta(masses, positions + nr1, velocities + nv1, dt / 2)
    if max([forces.magnitude(nr1[i] + nr2[i] - prev[i]) / forces.magnitude(nr1[i] + nr2[i]) for i in
            range(n_particles)]) > 1e-2:
        Nnr1, Nnv1, dtm1 = update_particles_recursive(masses, positions, velocities, dt / 2, nr1)
        Nnr2, Nnv2, dtm2 = update_particles_recursive(masses, positions + Nnr1.sum(axis=0),
                                                      velocities + Nnv1.sum(axis=0), dt / 2,
                                                      nr1 + nr2 - Nnr1.sum(axis=0))
        return np.concatenate((Nnr1, Nnr2)), np.concatenate((Nnv1, Nnv2)), np.concatenate((dtm1, dtm2))
    else:
        return np.array([nr1, nr2]), np.array([nv1, nv2]), np.array([dt / 2, dt / 2])


def calculate_kinetic_energies(masses, velocities):
    n = len(masses)
    tot = np.zeros(n)
    for i in range(n):
        tot[i] = 0.5 * masses[i] * forces.magnitude(velocities[i]) ** 2
    return tot


def calculate_potential_energies(masses, positions):
    n = len(masses)
    tot = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                tot[i] -= 6.67e-11 * masses[j] / forces.magnitude(positions[i] - positions[j])
    return tot


def animate(masses, positions, velocities, duration, speed, name):
    # make sure the three input arrays have consistent shapes
    n_particles, n_dimensions = positions.shape
    assert (velocities.shape == positions.shape)
    assert (len(masses) == n_particles)

    dt = 86400 * speed * 365 / (60 * 15)

    times_in_secs = np.arange(0, duration, dt)
    times_in_days = times_in_secs / 86400

    plt.style.use("dark_background")

    # Determine the framerate that results in one year in the simulation taking 15 seconds
    # oneyear = 15
    # dt_in_days = dt / 86400
    fps_ = 60  # round(speed * 365 / (oneyear * dt_in_days * samplingrate))

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
                              for i in range(n_particles)])

        # calculate the KE of each particle
        kinetic_energies = np.empty((n_particles, len(times_in_secs)))

        cmap = plt.get_cmap('plasma')
        new_cmap = truncate_colormap(cmap, 0.3, 1.0)

        # calculate the maximum distance of any of the particles from the origin
        max_distance_prev = max(
            [np.abs(np.max(distances)), np.abs(np.min(distances))])

        # # start a stopwatch
        # last = datetime.datetime.now()
        # with open('spacefacts.txt', 'r') as f:
        #     # read the lines
        #     lines = f.readlines()

        initial_sizes = np.clip(masses / max(masses) * 300, 10, 300)
        initial_scale = max_distance_prev * 1.1
        last_time = 0
        dtm = np.array([-1])
        npositions = np.array([positions])
        nvelocities = np.array([velocities])

        for i in tqdm(range(len(times_in_days))):
            time = times_in_secs[i]
            while time - last_time + dt * 16 > dtm[-1]:
                npositions, nvelocities, dtm = update_particles(masses, npositions[-1], nvelocities[-1], dt * 16)
                last_time += dt * 16

            stime = time - last_time + dt * 16

            for j in range(len(dtm) - 1):
                if dtm[j] <= stime <= dtm[j + 1]:
                    percent = (stime - dtm[j]) / (dtm[j + 1] - dtm[j])
                    positions = npositions[j] * (1 - percent) + npositions[j + 1] * percent
                    velocities = nvelocities[j] * (1 - percent) + nvelocities[j + 1] * percent
                    break

            kinetic_energies[:, i] = calculate_kinetic_energies(masses, velocities)

            distances = np.array([forces.magnitude(positions[k])
                                  for k in range(n_particles)])
            normed_distances = distances / np.max(distances)

            # this whole business keeps the axes on the same scale and (0,0,0) in the center
            bound = max([np.abs(np.max(distances)), np.abs(
                np.min(distances)), max_distance_prev])
            bound *= 1.1  # add a little padding

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
            sizes = initial_sizes * (initial_scale / bound) ** 2
            isometric.scatter(positions[:, 0], positions[:, 1],
                              positions[:, 2], s=sizes, c=normed_distances, cmap=new_cmap)

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
                             s=sizes, c=normed_distances, cmap=new_cmap, alpha=0.8)

            ke_2d.clear()
            for j in range(kinetic_energies.shape[0]):
                ke_2d.plot(times_in_days[:i], kinetic_energies[j, :i],
                           c=new_cmap(normed_distances[j]))
                ke_2d.set_xlim(times_in_days[0], times_in_days[-1])
                ke_2d.set_xlabel("Time")
                ke_2d.set_ylabel("Kinetic Energy")
                ke_2d.set_xticks([])
                ke_2d.set_yticks([])
            fig.suptitle(f'{name} at {times_in_days[i]/365:.2f} Years')
            fig.tight_layout()
            wri.grab_frame()

            # # if 10 seconds have passed, print a space fact
            # if i == 0:
            #     lastLineRead = 0
            # if (datetime.datetime.now() - last).seconds > 10:
            #     # if the line has not been read yet, read it
            #     if lastLineRead < len(lines):
            #         print(lines[lastLineRead])
            #         lastLineRead += 1
            #     last = datetime.datetime.now()

        wri.finish()
        print("Finishing up...")
        sleep(5)
        faketype("Video generated!")
        try:
            os.system(f'open "{filename}"')
        except OSError:
            faketype("Could not open video in default video player.")
