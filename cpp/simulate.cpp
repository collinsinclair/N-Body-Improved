#include "forces.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

// a function to read a text file and store data in arrays
// header: mass x y z vx vy vz
// data: mass x y z vx vy vz
// example:
// # mass (kg)   x (m)       y (m)       z (m)       vx (m/s)    vy (m/s)    vz (m/s)    
// +1.9890e+30 -9.9683e+10 +0.0000e+00 +0.0000e+00 -1.2667e+04 -1.9440e+04 +0.0000e+00
// +1.9890e+30 +9.9683e+10 +0.0000e+00 +0.0000e+00 -1.2667e+04 -1.9440e+04 +0.0000e+00
// +1.9890e+30 +0.0000e+00 +0.0000e+00 +0.0000e+00 +2.5334e+04 +3.8881e+04 +0.0000e+00
// returns number of particles

int read_data(string filename, int& n, double masses[], double positions[][3], double velocities[][3])
{
	// open file
	ifstream file(filename);
	if (!file) {
		cout << "Error: could not open file " << filename << endl;
		exit(1);
	}

	// read header
	string line;
	getline(file, line);

	// read data
	n = 0;
	while (getline(file, line)) {
		istringstream iss(line);
		double mass, x, y, z, vx, vy, vz;
		iss >> mass >> x >> y >> z >> vx >> vy >> vz;
		masses[n] = mass;
		positions[n][0] = x;
		positions[n][1] = y;
		positions[n][2] = z;
		velocities[n][0] = vx;
		velocities[n][1] = vy;
		velocities[n][2] = vz;
		n++;
	}
	return n;
}

int main(__attribute__((unused)) int argc, __attribute__((unused)) const char* argv[])
{
	int n = 30;
    double masses[n];
    double positions[n][3];
    double velocities[n][3];
    string path = "planetesimalDisk.txt";
    read_data(path, n, masses, positions, velocities);
    double dt = 0.01;
    double duration = 10;
    calculateTrajectories(masses, positions, velocities, duration, dt, n);
    return 0;
}