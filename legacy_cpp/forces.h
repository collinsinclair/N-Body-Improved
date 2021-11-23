//
// Created by Collin Sinclair on 11/6/21.
//

#ifndef FORCES
#define FORCES

#include<cmath>

// define gravitational constant
#define G 6.67408e-11

double forceMagnitude(double mi, double mj, double sep);
double magnitude(const double* vec, int n);
double* unitDirectionVector(const double* pos_a, const double* pos_b, int n);
double* forceVector(double mi, double mj, const double* pos_i, const double* pos_j, int n);
double* calculateForceVectors(double* masses, double positions[][3], int n, int m);
double** calculateTrajectories(double* masses, double positions[][3], double velocities[][3], double duration,
		double dt, int n);
double** evolveParticles(const double* masses, double positions[][3], double velocities[][3], double dt, int n);
#endif //FORCES
