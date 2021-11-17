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

    from src import rungekutta, systems

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


def make_videos_dir():
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


def faketype_intro():
    # clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    # roll the text onto the screen as if it's being typed by someone
    input(faketype(
        "Welcome to Snorto's N-Body Gravity Simulation! Press return to continue.", newline=False))
    input(faketype(
        "This program simulates the time evolution of various gravitational systems like the Sun and Moon, "
        "a planetesimal disk, or a star cluster. [return]", newline=False))
    input(faketype(
        "The simulation uses a Runge-Kutta Method, a iterative numerical method which we apply to the equations of "
        "motion of a system of particles. [return]", newline=False))
    input(faketype("""Each system has parameters that you can customize to modify the simulation, which include
    - the total duration (how long the simulation runs)
    - the number of bodies in the system
    - the scatter of starting velocities in the system
    - and more! [return]""", newline=False))
    # get current working directory
    cwd = os.getcwd()
    input(faketype("Each time you run a simulation, the program will store the resulting video in a 'videos' folder "
                   "that was created in the same folder you ran this program in: {}. [return]".format(cwd),
                   newline=False))


def faketype_system_menu():
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
    faketype("10. Who is Snorto?")
    faketype("11: Quit")
    faketype("--------------------")


def faketype_options(choose_n=False, default_n=0):
    faketype("""\nDURATION and SPEED
    The duration specifies the amount of time the simulation runs. The speed determines how fast the video plays. 1x
    speed corresponds to 15 seconds of
    video per year in simulation.""")
    choice = input("""Would you like to
    (1) use the defaults (1 year, 1x speed) or
    (2) enter your own?
Enter a 1 or 2: """)
    while choice != "1" and choice != "2":
        faketype("Invalid selection. Enter 1 or 2: ", newline=False)
        choice = input()
    if choice == "1":
        duration = 365 * 24 * 60 * 60  # 1 year
        speed = 1
    else:
        duration = float(input("Enter a duration in years: ")
                         ) * (365 * 60 * 60 * 24)
        speed = int(input("Enter a speed multiplier: "))
    if choose_n:
        faketype(f"""\nNUMBER OF PARTICLES""")
        n_choice = input(f"""Would you like to
    (1) use the default number of particles ({default_n}) or
    (2) enter your own?
Enter a 1 or a 2: """)
        while n_choice != "1" and n_choice != "2":
            n_choice = input("Invalid selection. Enter a 1 or as 2: ")
        if n_choice == "1":
            n = 0
        else:
            n = int(input("Enter the number of particles: "))
    else:
        n = 0
    return duration, speed, n


def faketype_velocity_scatter(default=0):
    faketype(f"""\nVELOCITY SCATTER
    The velocity scatter specifies the width of the Gaussian distribution from which 
    to draw the initial velocities of the particles. The higher the value, the more spread out the velocities will 
    be.""")
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


def faketype_mass_ratios(default=0.0):
    faketype(f"""\nMASS RATIO
    The evolution of the system depends on the ratio of mass each planetesimal to the mass of the central star. At very
    small values, the gravity is totally dominated by the central star; at larger values (above about 1e-6, roughly an
    Earth mass per particle), the orbits may start to go unstable due to the interactions between the particles.""")
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


def faketype_z_velocities(default=0):
    faketype(f"""\nZ VELOCITY
    The z velocity of the particles is the vertical component of their velocity. The z 
    velocity is the component of the velocity in the direction of the z axis.""")
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


def faketype_max_mass(default=0.0):
    m_sun = 1.98892e30  # kg
    faketype("""\nMAXIMUM MASS
    The maximum mass of the particles in the cluster is 0.01 solar masses by default.""")
    choice = input(f"""Would you like to
    (1) use the default ({default / m_sun:.2f} solar masses) or
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


def simulate_sun_earth():
    masses, positions, velocities = systems.SunEarth()
    duration, speed, n = faketype_options()
    return masses, positions, velocities, duration, speed


def simulate_sun_earth_moon():
    masses, positions, velocities = systems.SunEarthMoon()
    duration, speed, n = faketype_options()
    return masses, positions, velocities, duration, speed


def simulate_kepler_16():
    masses, positions, velocities = systems.Kepler16()
    duration, speed, n = faketype_options()
    return masses, positions, velocities, duration, speed


def simulate_random_cube():
    duration, speed, n = faketype_options(choose_n=True, default_n=30)
    if n == 0:
        n = 30
    velocity_scatter = faketype_velocity_scatter(2000)
    if velocity_scatter == 0:
        velocity_scatter = 2000
    masses, positions, velocities = systems.randomCube(n, velocity_scatter)
    return masses, positions, velocities, duration, speed


def simulate_uniform_cube():
    duration, speed, n = faketype_options(True, 16)
    if n == 0:
        n = 16
    velocity_scatter = faketype_velocity_scatter(5000)
    if velocity_scatter == 0:
        velocity_scatter = 5000
    masses, positions, velocities = systems.uniformCube(n, velocity_scatter)
    return masses, positions, velocities, duration, speed


def simulate_pythagorean():
    masses, positions, velocities = systems.pythagorean()
    duration, speed, n = faketype_options()
    return masses, positions, velocities, duration, speed


def simulate_figure_8():
    masses, positions, velocities = systems.figure8()
    duration, speed, n = faketype_options()
    return masses, positions, velocities, duration, speed


def simulate_planetesimal_disk():
    duration, speed, n = faketype_options(True, 30)
    if n == 0:
        n = 30
    mass_ratio = faketype_mass_ratios(1e-10)
    if mass_ratio == 0:
        mass_ratio = 1e-10
    z_velocity = faketype_z_velocities(1000)
    if z_velocity == 0:
        z_velocity = 1000
    masses, positions, velocities = systems.planetesimalDisk(
        n, mass_ratio, z_velocity)
    return masses, positions, velocities, duration, speed


def simulate_tiny_cluster():
    duration, speed, n = faketype_options(True, 20)
    if n == 0:
        n = 20
    max_mass = faketype_max_mass(0.01 * 1.989e30)
    if max_mass == 0:
        max_mass = 0.01 * 1.989e30
    masses, positions, velocities = systems.tinyCluster(n, max_mass)
    return masses, positions, velocities, duration, speed


def goodbye():
    # clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    # get path to video directory
    video_dir = os.path.join(os.getcwd(), "videos")
    # get size of video directory
    size = 0
    for path, dirs, files in os.walk(video_dir):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    faketype(
        f"Your videos are stored in {video_dir}, and they take up {size / 1e6:.1f} MB.")
    faketype("Thank you for using the simulation!")


def select_system():
    system_options = ["SunEarth", "SunEarthMoon", "Kepler16", "RandomCube",
                      "UniformCube", "Pythagorean", "Figure8", "PlanetesimalDisk", "TinyCluster"]
    faketype_system_menu()
    system = input("Please select a system: ")
    # attempt to convert the input to an integer
    valid_choice = False
    while not valid_choice:
        try:
            system = int(system)
            if system < 1 or system > 11:
                system = input(
                    "Invalid choice. Please enter a number between 1 and 9: ")
            else:
                valid_choice = True
        except ValueError:
            system = input(
                "Invalid choice. Please enter a number between 1 and 9: ")
    if system == 1:
        # clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Sun-Earth")
        return simulate_sun_earth(), system_options[system - 1]
    elif system == 2:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Sun-Earth-Moon")
        return simulate_sun_earth_moon(), system_options[system - 1]
    elif system == 3:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Kepler-16")
        return simulate_kepler_16(), system_options[system - 1]
    elif system == 4:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Random Cube")
        return simulate_random_cube(), system_options[system - 1]
    elif system == 5:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Uniform Cube")
        return simulate_uniform_cube(), system_options[system - 1]
    elif system == 6:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Pythagorean")
        return simulate_pythagorean(), system_options[system - 1]
    elif system == 7:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Figure 8")
        return simulate_figure_8(), system_options[system - 1]
    elif system == 8:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Planetesimal Disk")
        return simulate_planetesimal_disk(), system_options[system - 1]
    elif system == 9:
        os.system('cls' if os.name == 'nt' else 'clear')
        faketype("--------------------")
        faketype("Tiny Cluster")
        return simulate_tiny_cluster(), system_options[system - 1]
    elif system == 10:
        print("""SNORTO
        The year was 2019. Collin, Brandon, and some others (our roommates and Collin's girlfriend) were headed to a
        crag in Boulder Canyon called Solaris for a fun day of rock climbing. In the car on the way, Brandon asked, 
        "Where are we going again? Snor-toes?" We all had a good laugh, and Snorto has referred to the inhabitants Unit
        3311 ever since.""")
    elif system == 11:
        goodbye()
        sys.exit(0)


def main():
    make_videos_dir()
    faketype_intro()
    cont = True
    while cont:
        # clear the screen
        os.system('cls' if os.name == 'nt' else 'clear')
        try:
            (masses, positions, velocities, duration, speed), name = select_system()
        except TypeError:
            input('[return]')
            continue
        faketype("--------------------")
        rungekutta.animate(masses, positions, velocities,
                           duration, speed, name)
        cont = input("Would you like to run another simulation? (y/n) ")
        while cont != "y" and cont != "n":
            cont = input("Would you like to run another simulation? (y/n) ")
        if cont == "n":
            goodbye()
            cont = False


if __name__ == "__main__":
    main()
