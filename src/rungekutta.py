import datetime
import os
import sys
from time import sleep

import matplotlib.animation as ani
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from cpp import Simulator


def magnitude(vec):
    """Calculate the magnitude of a vector in numpy array form"""
    return np.sqrt(np.sum(vec ** 2))


def truncate_color_map(color_map, min_val=0.0, max_val=1.0, n=100):
    """Truncate the color map color_map to eliminate dark colors"""
    new_color_map = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=color_map.name, a=min_val,
                                                b=max_val),
            color_map(np.linspace(min_val, max_val, n)))
    return new_color_map


def fake_type(words, speed=0.001, newline=True):
    """Print words slowly to emulate typing"""
    for char in words:
        sleep(speed)
        print(char, end="")
        sys.stdout.flush()
    if newline:
        print("")
    return ""


def calculate_kinetic_energies(masses, velocities):
    """Calculate the kinetic energy of the particles"""
    n = len(masses)
    tot = np.zeros(n)
    for i in range(n):
        tot[i] = 0.5 * masses[i] * magnitude(velocities[i]) ** 2
    return tot


def calculate_potential_energies(masses, positions):
    """Calculate the potential energy of the particles"""
    n = len(masses)
    tot = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                tot[i] -= (6.67e-11 * masses[j]
                           / magnitude(positions[i] - positions[j]))
    return tot


def calculate_range(distances):
    """Calculate the 75th percentile of a numpy array and pad it"""
    return np.percentile(distances, 75) * 1.1


def animate(masses, positions, velocities, duration, speed, name):
    """Animate the system provided and save the video"""
    # make sure the three input arrays have consistent shapes
    n_particles, n_dimensions = positions.shape
    assert (velocities.shape == positions.shape)
    assert (len(masses) == n_particles)

    fps = 60

    # Calculate the time step required to meet desired fps
    dt = 86400 * speed * 365 / (fps * 15)

    # Create arrays of all times in the duration
    times_in_secs = np.arange(0, duration, dt)
    times_in_days = times_in_secs / 86400

    plt.style.use("dark_background")

    # Determine the frame rate that results in one year in the simulation
    # taking 15 seconds

    # Set up the figure
    wri = ani.FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(30, 10))
    isometric = fig.add_subplot(132, projection='3d')
    ke_2d = fig.add_subplot(396)
    xz_plane = fig.add_subplot(131)
    xy_plane = fig.add_subplot(133)

    # Set the filename and include date/time
    filename = ('./videos/' + name + '_' +
                datetime.datetime.now().strftime(
                        "%Y-%m-%d-%H-%M-%S") + '.mp4')

    # noinspection SpellCheckingInspection
    with wri.saving(fig, filename, 100):
        fake_type("Calculating trajectories and generating video...")

        # calculate the distance of each particle to the origin
        distances = np.array([magnitude(positions[i])
                              for i in range(n_particles)])
        normed_distances = distances / np.max(distances)

        # calculate the kinetic energy of each particle
        kinetic_energies = np.empty((n_particles, len(times_in_secs)))
        kinetic_energies[:, 0] = calculate_kinetic_energies(masses,
                                                            velocities)

        color_map = plt.get_cmap('plasma')
        new_color_map = truncate_color_map(color_map, 0.3, 1.0)

        # calculate the maximum distance of any of the particles from the
        # origin
        range_prev = calculate_range(distances)
        max_distance_prev = np.max(np.abs(distances))

        # Set initial bounds and sizes for scaling later
        initial_sizes = np.clip(masses / max(masses) * 300, 10, 300)
        initial_scale = max_distance_prev * 1.1
        bound = initial_scale
        sizes = initial_sizes

        # Create new Simulator instance
        simulator = Simulator.simulator(masses.tolist(),
                                        positions.flatten().tolist(),
                                        velocities.flatten().tolist(),
                                        dt, n_particles)

        # Set up 3D plot
        isometric.set_xlabel("x")
        isometric.set_ylabel("y")
        isometric.set_zlabel("z")
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
        isometric.set_xlim(-bound, bound)
        isometric.set_ylim(-bound, bound)
        isometric.set_zlim(-bound, bound)
        iso = isometric.scatter(positions[:, 0], positions[:, 1],
                                positions[:, 2], s=sizes, c=normed_distances,
                                cmap=new_color_map)

        # Set up X-Z plane plot
        xz_plane.set_xlabel("x")
        xz_plane.set_ylabel("z")
        xz_plane.set_xlim(-bound, bound)
        xz_plane.set_ylim(-bound, bound)
        xz_plane.grid(False)
        xz_plane.facecolor = 'black'
        xz_plane.set_xticks([])
        xz_plane.set_yticks([])
        xz = xz_plane.scatter(positions[:, 0], positions[:, 2],
                              s=sizes, c=normed_distances, cmap=new_color_map,
                              alpha=0.8)

        # Set up X-Y plane plot
        xy_plane.set_xlabel("x")
        xy_plane.set_ylabel("y")
        xy_plane.set_xlim(-bound, bound)
        xy_plane.set_ylim(-bound, bound)
        xy_plane.grid(False)
        xy_plane.facecolor = 'black'
        xy_plane.set_xticks([])
        xy_plane.set_yticks([])
        xy = xy_plane.scatter(positions[:, 0], positions[:, 1],
                              s=sizes, c=normed_distances, cmap=new_color_map,
                              alpha=0.8)

        # Set up kinetic energy plot
        kes = [ke_2d.plot(times_in_days, kinetic_energies[j],
                          c=new_color_map(normed_distances[j])) for j in
               range(n_particles)]
        ke_2d.set_xlim(times_in_days[0], times_in_days[-1])
        ke_2d.set_xlabel("Time")
        ke_2d.set_ylabel("Kinetic Energy")
        ke_2d.set_xticks([])
        ke_2d.set_yticks([])

        fig.tight_layout()

        # Loop over all times in duration
        for i in tqdm(range(len(times_in_days))):
            # Update system
            simulator.stepForward()

            # Convert positions and velocities from C shape to python shape
            positions = np.array(simulator.getPositions()).reshape(
                    (n_particles, n_dimensions))
            velocities = np.array(simulator.getVelocities()).reshape(
                    (n_particles, n_dimensions))

            kinetic_energies[:, i] = calculate_kinetic_energies(masses,
                                                                velocities)

            distances = np.array([magnitude(positions[k])
                                  for k in range(n_particles)])
            normed_distances = distances / np.max(distances)

            # this whole business keeps the axes on the same scale and
            # (0, 0,0) in the center
            max_distance_prev = max(max_distance_prev,
                                    np.max(np.abs(distances)))
            range_prev = max(range_prev,
                             calculate_range(distances))
            bound = max(bound, min(range_prev,
                                   max_distance_prev * 1.1))
            sizes = initial_sizes * (initial_scale / bound) ** 2

            # Update bounds and particle locations
            isometric.set_xlim(-bound, bound)
            isometric.set_ylim(-bound, bound)
            isometric.set_zlim(-bound, bound)

            iso._offsets3d = (positions[:, 0],
                              positions[:, 1],
                              positions[:, 2])
            iso.set_array(normed_distances)
            iso._sizes3d = sizes

            xz_plane.set_xlim(-bound, bound)
            xz_plane.set_ylim(-bound, bound)

            xz.set_offsets(np.hstack((positions[:, 0, np.newaxis],
                                      positions[:, 2, np.newaxis])))
            xz.set_array(normed_distances)
            xz._sizes = sizes

            xy_plane.set_xlim(-bound, bound)
            xy_plane.set_ylim(-bound, bound)
            xy.set_offsets(np.hstack((positions[:, 0, np.newaxis],
                                      positions[:, 1, np.newaxis])))
            xy.set_array(normed_distances)
            xy._sizes = sizes

            # Update kinetic energy plot
            for j in range(n_particles):
                kes[j][0].set_data(times_in_days[:i],
                                   kinetic_energies[j, :i])
                # TODO this broke with the new drawing method
                kes[j][0].set_color(new_color_map(normed_distances[j]))

            fig.suptitle(f'{name} at {times_in_days[i] / 365:.2f} Years')
            wri.grab_frame()

        # Save video
        wri.finish()
        fake_type("Finishing up...")
        sleep(5)
        fake_type("Video generated!")

        # Try to open the newly generated video
        try:
            os.system(f'open "{filename}"')
        except OSError:
            fake_type("Could not open video in default video player.")
