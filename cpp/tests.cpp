//
// Created by Collin Sinclair on 11/6/21.
//

#include "tests.h"
#include <iostream>

bool test_forceMagnitude()
{
	double mEarth = 5.972e24;
	double mPerson = 70;
	double radiusEarth = 6.371e6;
	double force = forceMagnitude(mEarth, mPerson, radiusEarth);
	// assert that force is within 3% of expected value
	double expected = 683.936;
	double tolerance = 0.01;
	return (force>=expected*(1-tolerance)) && (force<=expected*(1+tolerance));
}
bool test_magnitude()
{
	double vec[3] = {3.0, 4.0, 0.0};
	double mag = magnitude(vec, 3);
	double expected = 5.0;
	double tolerance = 0.01;
	return (mag>=expected*(1-tolerance)) && (mag<=expected*(1+tolerance));
}
bool test_unitDirectionVector()
{
	double someplace[3] = {3.0, 2.0, 5.0};
	double somewhere[3] = {1.0, -4.0, 8.0};
	double expected[3] = {-0.28571429, -0.85714286, 0.42857143};
	double tolerance = 0.01;
	double* unitDirVec = unitDirectionVector(someplace, somewhere, 3);
	bool result = true;
	for (int i = 0; i<3; i++) {
		// percent difference between expected and unitDirVec
		double diff = fabs(unitDirVec[i]-expected[i])/expected[i];
		result = result && (diff<tolerance);
	}
	delete[] unitDirVec;
	return result;
}
bool test_forceVector()
{
	double mEarth = 6.0e24;
	double mPerson = 70;
	double radiusEarth = 6.4e6;
	double centerEarth[3] = {0, 0, 0};
	double surfaceEarth[3] = {0, 0, radiusEarth};
	double expected[3] = {0,
						  0,
						  683.93554688};
	double* forceVec = forceVector(mEarth, mPerson, centerEarth, surfaceEarth, 3);
	bool result = true;
	double tolerance = 0.01;
	for (int i = 0; i<3; i++) {
		// percent difference between expected and forceVec
		result &= (forceVec[i]>=expected[i]*(1-tolerance)) && (forceVec[i]<=expected[i]*(1+tolerance));
	}
	delete[] forceVec;
	return result;
}
bool test_calculateForceVectors()
{
	double au = 149597870700;
	double masses[5] = {1.0e24, 40.0e24, 50.0e24, 30.0e24, 2.0e24};
	double positions[5][3] = {
			{0.5*au, 2.6*au, 0.05*au},
			{0.8*au, 9.1*au, 0.1*au},
			{-4.1*au, -2.4*au, 0.8*au},
			{10.7*au, 3.7*au, 0},
			{-2*au, -1.9*au, -0.4*au}};
	double expected[5][3] = {
			{-1.3e+15, 3.8e+14, 3.5e+14},
			{9.2e+15, -5.3e+16, 1.8e+15},
			{7.5e+16, 5.4e+16, -2.7e+16},
			{-4.2e+16, 6.4e+15, 1.1e+15},
			{-4.0e+16, -7.5e+15, 2.4e+16}};
	double* forceVectors = calculateForceVectors(masses, positions, 5, 3);
	bool result = true;
	double tolerance = 0.015;
	for (int i = 0; i<5; i++) {
		for (int j = 0; j<3; j++) {
			double diff = fabs(forceVectors[3*i+j]-expected[i][j])/expected[i][j];
			result = result && (diff<tolerance);
		}
	}
	delete[] forceVectors;
	return result;
}