//
// Created by Collin Sinclair on 11/6/21.
//

#include "forces.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

double forceMagnitude(double mi, double mj, double sep)
{
	return G*mi*mj/(sep*sep);
}

double magnitude(const double* vec, int n)
{
	/*
	Compute magnitude of any vector with an arbitrary number of elements.

	Parameters
	----------
	vec : array

	Returns
	-------
	magnitude : float
		The magnitude of that vector.
	*/
	double mag = 0;
	for (int i = 0; i<n; i++) {
		mag += vec[i]*vec[i];
	}
	return sqrt(mag);
}

double* unitDirectionVector(const double* pos_a, const double* pos_b, int n)
{
	/*
	Create unit direction vector from pos_a to pos_b

	Parameters
	----------
	pos_a, pos_b : two arrays
		Any two vectors

	Returns
	-------
	unit direction vector : one array (same size input vectors)
		The unit direction vector from pos_a toward pos_b
	*/
	auto* dir = new double[n];    // direction vector
	for (int i = 0; i<n; i++) // iterate over each dimension
	{
		dir[i] = pos_b[i]-pos_a[i]; // compute direction vector
	}
	double mag = magnitude(dir, n);
	for (int i = 0; i<n; i++) // iterate over each dimension
	{
		dir[i] /= mag; // normalize direction vector
	}
	return dir;
}

double* forceVector(double mi, double mj, const double pos_i[3], const double pos_j[3], int n)
{
	/*
	Compute gravitational force vector exerted on particle i by particle j.

	Parameters
	----------
	mi, mj : floats
		Particle masses, in kg.
	pos_i, pos_j : arrays
		Particle positions in cartesian coordinates, in m.

	Returns
	-------
	forceVec : array
		Components of gravitational force vector, in N.
	*/
	// compute pos_i - pos_j
	auto* r = new double[n];
	for (int i = 0; i<n; i++) {
		r[i] = pos_i[i]-pos_j[i];
	}
	// compute magnitude of r
	double sep = magnitude(r, n);
	// compute the magnitude of the force
	double force = forceMagnitude(mi, mj, sep);
	// compute the unit direction vector
	auto* dir = unitDirectionVector(pos_i, pos_j, n);
	// compute the force vector
	auto* forceVec = new double[n];
	for (int i = 0; i<n; i++) {
		forceVec[i] = force*dir[i];
	}
	return forceVec;
}

double* calculateForceVectors(const double* masses, double* positions[3], int n)
{
	/*
	Compute net gravitational force vectors on particles,
	given a list of masses and positions for all of them.

	Parameters
	----------
	masses : array of floats
		Particle masses, in kg.
	positions : array of 3-element arrays
		Particle positions in cartesian coordinates, in meters,
		in the same order as the masses are listed. Each element
		in the list (a single particle's position) should be a
		3-element array, referring to its X, Y, Z position.

	Returns
	-------
	forceVectors : list of 3-element arrays
		A list containing the net force vectors for each particles.
		Each element in the list is a 3-element array that
		represents the net 3D force acting on a particle, after summing
		over the individual force vectors induced by every other particle.
*/
	auto* forceVectors = new double[n*3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			forceVectors[i*3+j] = 0;
		}
	}
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<n; j++) {
			if (i!=j) {
				auto* force = forceVector(masses[i], masses[j], positions[i], positions[j], 3);
				for (int k = 0; k<3; k++) {
					forceVectors[i*3+k] += force[k];
				}
			}
		}
	}
	return forceVectors;
}

double** evolveParticles(const double* masses, double positions[][3], double velocities[][3], double dt, int n)
{
	/*
    Evolve particles in time via leap-frog integrator scheme. This function
    takes masses, positions, velocities, and a time step dt as

    Parameters
    ----------
    masses : array
        1-D array containing masses for all particles, in kg
        It has length N, where N is the number of particles.
    positions : array
        2-D array containing (x, y, z) positions for all particles.
        Shape is (N, 3) where N is the number of particles.
    velocities : array
        2-D array containing (x, y, z) velocities for all particles.
        Shape is (N, 3) where N is the number of particles.
    dt : float
        Evolve system for time dt (in seconds).

    Returns
    -------
    Updated particle positions and particle velocities, each being a 2-D
    array with shape (N, 3), where N is the number of particles.
*/
	// copy positions and velocities
	auto* startingPositions = new double[n][3];
	auto* startingVelocities = new double[n][3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			startingPositions[i][j] = positions[i][j];
			startingVelocities[i][j] = velocities[i][j];
		}
	}
	// calculate net force vectors on all particles, at the starting position
	double* startingForces = calculateForceVectors(masses, reinterpret_cast<double**>(startingPositions), n);
	// calculate the acceleration due to gravity, at the starting position
	auto* startingAccelerations = new double[n*3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			startingAccelerations[i*3+j] = startingForces[i*3+j]/masses[i];
		}
	}
	// nudge = startingVelocities*dt + 1/2*startingAccelerations*dt^2
	auto* nudge = new double[n*3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			nudge[i*3+j] = startingVelocities[i][j]*dt+0.5*startingAccelerations[i*3+j]*dt*dt;
		}
	}
	// newPositions = startingPositions + nudge
	auto* newPositions = new double[n][3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			newPositions[i][j] = startingPositions[i][j]+nudge[i*3+j];
		}
	}

	// calculate net force vectors on all particles, at the ending position
	double* endingForces = calculateForceVectors(masses, reinterpret_cast<double**>(newPositions), n);
	// calculate the acceleration due to gravity, at the ending position
	auto* endingAccelerations = new double[n*3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			endingAccelerations[i*3+j] = endingForces[i*3+j]/masses[i];
		}
	}
	// newVelocities = startingVelocities + 1/2*(startingAccelerations + endingAccelerations)*dt
	auto* newVelocities = new double[n][3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			newVelocities[i][j] =
					startingVelocities[i][j]+0.5*(startingAccelerations[i*3+j]+endingAccelerations[i*3+j])*dt;
		}
	}
	// create an array of pointers to the new positions and velocities
	auto* newPositionsAndVelocities = new double* [2];
	newPositionsAndVelocities[0] = reinterpret_cast<double*>(newPositions);
	newPositionsAndVelocities[1] = reinterpret_cast<double*>(newVelocities);
	return newPositionsAndVelocities;
}

double** calculateTrajectories(double* masses, double positions[][3], double velocities[][3], double duration,
		double dt, int n)
{
	/*
	Compute net gravitational force vectors on particles,
	given a list of masses and positions for all of them.

	Parameters
	----------
	masses : array of floats
		Particle masses, in kg.
	positions : array of 3-element arrays
		Particle positions in cartesian coordinates, in meters,
		in the same order as the masses are listed. Each element
		in the list (a single particle's position) should be a
		3-element array, referring to its X, Y, Z position.

	Returns
	-------
	forceVectors : list of 3-element arrays
		A list containing the net force vectors for each particles.
		Each element in the list is a 3-element array that
		represents the net 3D force acting on a particle, after summing
		over the individual force vectors induced by every other particle.
	*/

	// copy the positions array
	auto* positions_copy = new double[n][3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			positions_copy[i][j] = positions[i][j];
		}
	}

	// copy the velocities array
	auto* starting_velocities = new double[n][3];
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<3; j++) {
			starting_velocities[i][j] = velocities[i][j];
		}
	}

	// create a times array from 0 to duration in steps of dt
	auto* times = new double[(int)(duration/dt)+1];
	for (int i = 0; i<(int)(duration/dt)+1; i++) {
		times[i] = i*dt;
	}

	// size of times array
	int n_times = (int)(duration/dt)+1;

	// initialize an n by 3 by ntimes array to store the positions
	auto* positions_array = new double* [n];
	for (int i = 0; i<n; i++) {
		positions_array[i] = new double[3*n_times];
	}

	// initialize an n by 3 by ntimes array to store the velocities
	auto* velocities_array = new double* [n];
	for (int i = 0; i<n; i++) {
		velocities_array[i] = new double[3*n_times];
	}
//#pragma omp parallel for
	for (int i = 0; i<n_times; ++i) {
		// compute updated positions and velocities
		auto* new_positions_and_velocities = evolveParticles(masses, positions_copy, starting_velocities, dt, n);
		// copy the new positions and velocities into the arrays
		for (int j = 0; j<n; j++) {
			for (int k = 0; k<3; k++) {
				positions_array[j][i*3+k] = new_positions_and_velocities[0][j*3+k];
				velocities_array[j][i*3+k] = new_positions_and_velocities[1][j*3+k];
			}
		}
		// update the positions and velocities
		for (int j = 0; j<n; j++) {
			for (int k = 0; k<3; k++) {
				positions_copy[j][k] = new_positions_and_velocities[0][j*3+k];
				starting_velocities[j][k] = new_positions_and_velocities[1][j*3+k];
			}
		}
		// delete the new positions and velocities
		//delete[] new_positions_and_velocities;
 		// print progress bar to the console
		if (i%(int)(duration/dt/10)==0) {  
			std::cout << "Progress: " << i*100/(int)(duration/dt) << "%" << std::endl;  
		}
	}
	// write times, positions, and velocities to a file
	std::ofstream file;
	file.open("trajectories.txt");
	file << n_times << std::endl;
	for (int i = 0; i<n_times; i++) {
		file << times[i] << " ";
		for (int j = 0; j<n; j++) {
			for (int k = 0; k<3; k++) {
				file << positions_array[j][i*3+k] << " ";
			}
		}
		for (int j = 0; j<n; j++) {
			for (int k = 0; k<3; k++) {
				file << velocities_array[j][i*3+k] << " ";
			}
		}
		file << std::endl;
	}
	file.close();

	// create an array of pointers to the times, positions and velocities arrays
	auto* trajectories = new double* [3];
	trajectories[0] = times;
	trajectories[1] = reinterpret_cast<double*>(positions_array);
	trajectories[2] = reinterpret_cast<double*>(velocities_array);
	return trajectories;
}
