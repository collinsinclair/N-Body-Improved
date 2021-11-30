#include <math.h>
#include <iostream>
#include <ctime>
#include <string>

using namespace std;

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Add two n x 3 arrays together
double *sum_tensors(double *tensor_1, double *tensor_2, int n){
	double *new_tensor = new double[n*3];
	for(int i = 0; i<n*3; i++){
		new_tensor[i] = tensor_1[i] + tensor_2[i];
	}
	return new_tensor;
}

// Multiply every element of an n x 3 array by a scalar
double *scalar_mult_tensor(double *tensor, double scalar, int n){
	double *new_tensor = new double[n*3];
	for(int i = 0; i<n*3; i++){
		new_tensor[i] = tensor[i] * scalar;
	}
	return new_tensor;
}

// Add an n x 3 array scaled by a scalar to another n x 3 array
// This function is included to minimize copy operations
double *sum_scalar_mult_tensor(double *tensor, double *tensor_2, double scalar, int n){
	double *new_tensor = new double[n*3];
	for(int i = 0; i<n*3; i++){
		new_tensor[i] = tensor[i] + tensor_2[i] * scalar;
	}
	return new_tensor;
}

// Copy an n x 3 array to a new location
double *copy_tensor(double *original, int n){
	double *new_tensor = new double[n*3];
	for(int i = 0; i<n*3; i++){
		new_tensor[i] = original[i];
	}
	return new_tensor;
}

// Timer class to simplify optimization testing
// Create a new timer, t.  This will start the time.
// Call t.time([task]) to print how long it has been since
// the time started.  Optionally include a task to print
// the task as well.  Call t.start() to reset the timer.
class Timer{
public:
	double start_time;
	Timer(){
		start_time = clock();
	}
	void start(){
		start_time = clock();
	}
	void time(){
		cout << "Time since start: " << (clock()-start_time) << endl;
	}
	void time(string task){
		cout << "Time to " << task << ": " << (clock()-start_time) << endl;
	}
};

// Class to store data calculated in one simulation (broken into subdivisions)
class SimStep{
public:
	int n;
	int depth;
	double **positions_array;
	double **velocities_array;
	double *times_array;

	SimStep(){
		depth = 0;
	}

    // Delete the arrays containing position, velocity, and time data,
    // but don't delete the positions, velocities and times.
    // This is done to allow reuse of this data to reduce copy operations.
	~SimStep(){
		if(depth > 0){
			delete[] positions_array;
			delete[] velocities_array;
			delete[] times_array;
			depth = 0;
		}
	}

    // Delete all position and velocity arrays stored in SimStep
    // and delete the arrays containing these arrays.  Only use
    // when you are sure that the data will not be used again.
	void obliterate(){
		if(depth > 0){
			for(int i = 0; i<depth; i++){
				delete[] positions_array[i];
				delete[] velocities_array[i];
			}
			delete[] positions_array;
			delete[] velocities_array;
			delete[] times_array;
			depth = 0;
		}
	}

	SimStep(const SimStep &other){
		n = other.n;
		depth = other.depth;
		if(depth > 0){
			positions_array = new double*[depth];
			velocities_array = new double*[depth];
			times_array = new double[depth];
			for(int i = 0; i<depth; i++){
				positions_array[i] = other.positions_array[i];
				velocities_array[i] = other.velocities_array[i];
				times_array[i] = other.times_array[i];
			}
		}
	}

	SimStep(double **positions_array, double **velocities_array, double *times_array, int depth, int n){
		this->positions_array = positions_array;
		this->velocities_array = velocities_array;
		this->times_array = times_array;
		this->depth = depth;
		this->n = n;
	}

	SimStep(double *positions, double *velocities, double time, int n){
		positions_array = new double*[1];
		positions_array[0] = positions;
		velocities_array = new double*[1];
		velocities_array[0] = velocities;
		times_array = new double[1];
		times_array[0] = time;
		depth = 1;
		this->n = n;
	}

	SimStep(double *positions_0, double *positions_1, double *velocities_0, double *velocities_1, double time_0, double time_1, int n){
		positions_array = new double*[2];
		positions_array[0] = positions_0;
		positions_array[1] = positions_1;
		velocities_array = new double*[2];
		velocities_array[0] = velocities_0;
		velocities_array[1] = velocities_1;
		times_array = new double[2];
		times_array[0] = time_0;
		times_array[1] = time_1;
		depth = 2;
		this->n = n;
	}

	SimStep operator+(SimStep other){
		double **new_positions = new double*[depth+other.depth];
		double **new_velocities = new double*[depth+other.depth];
		double *new_times = new double[depth+other.depth];
		for(int i = 0; i<depth; i++){
			new_positions[i] = positions_array[i];
			new_velocities[i] = velocities_array[i];
			new_times[i] = times_array[i];
		}
		for(int i = 0; i<other.depth; i++){
			new_positions[i+depth] = other.positions_array[i];
			new_velocities[i+depth] = other.velocities_array[i];
			new_times[i+depth] = other.times_array[i];
		}
		return SimStep(new_positions, new_velocities, new_times, depth+other.depth, n);
	}

    // Calculate the positions and velocities time t into the SimStep
    // and store the values in the pointers passed in
	void update(double t, double *positions, double *velocities, int n){
		for(int i = 0; i<depth - 1; i++){
			if(times_array[i] <= t && times_array[i+1] >= t){
				double fraction = (t - times_array[i])/(times_array[i+1] - times_array[i]);
				for(int j = 0; j<n*3; j++){
					positions[j] = positions_array[i][j]*(1-fraction) + positions_array[i+1][j]*(fraction);
					velocities[j] = velocities_array[i][j]*(1-fraction) + velocities_array[i+1][j]*(fraction);
				}
				break;
			}
		}
	}

    //Print all time steps stored in SimStep for debug
	void printTimes(){
		cout << "T: ";
		for(int i = 0; i<depth; i++){
			cout << times_array[i] << ", ";
		}
		cout << endl;
	}

	void operator=(const SimStep &other){
		if(depth>0){
			delete[] positions_array;
			delete[] velocities_array;
			delete[] times_array;
		}
		n = other.n;
		depth = other.depth;
		if(depth > 0){
			positions_array = new double*[depth];
			velocities_array = new double*[depth];
			times_array = new double[depth];
			for(int i = 0; i<depth; i++){
				positions_array[i] = other.positions_array[i];
				velocities_array[i] = other.velocities_array[i];
				times_array[i] = other.times_array[i];
			}
		}
	}
};

// Calculate the array of acceleration vectors for the given system
// using Newton's law of gravitation
double *calculate_acceleration(double *masses, double *positions, int n){
	double G = 6.67e-11;
    double *acceleration = new double[n*3];
    for(int i = 0; i<n*3; i++){
    	acceleration[i] = 0;
    }
    for(int i = 0; i<n; i++){
	    for(int j = i+1; j<n; j++){
        	double r = 0;
        	for(int k = 0; k<3; k++){
        		r += pow(positions[i*3+k]-positions[j*3+k], 2);
        	}
        	double magnitude = pow(r, -1.5);
        	for(int k = 0; k<3; k++){
        		double directional_magnitude = magnitude*(positions[j*3+k] - positions[i*3+k]);
        		acceleration[i*3+k] += masses[j]*directional_magnitude;
        		acceleration[j*3+k] -= masses[i]*directional_magnitude;
        	}
    	}
    }
    for(int i = 0; i<n*3; i++){
    	acceleration[i] *= G;
    }
    return acceleration;
}

// Perform Runge--Kutta 4th order numerical approximation of 2nd order
// position -> velocity -> acceleration equations from Newton's 2nd law
double **runge_kutta(double *masses, double *positions, double *velocities, int n, double time_step){
    double *velocity_estimate_0 = calculate_acceleration(masses, positions, n);

    double *position_estimate_0 = velocities;

    double *state_1_positions = sum_scalar_mult_tensor(positions, position_estimate_0, time_step/2.0, n);
    double *velocity_estimate_1 = calculate_acceleration(masses, state_1_positions, n);
    delete[] state_1_positions;

    double *position_estimate_1 = sum_scalar_mult_tensor(velocities, velocity_estimate_0, time_step/2.0, n);

    double *state_2_positions = sum_scalar_mult_tensor(positions, position_estimate_1, time_step/2.0, n);
    double *velocity_estimate_2 = calculate_acceleration(masses, state_2_positions, n);
    delete[] state_2_positions;

    double *position_estimate_2 = sum_scalar_mult_tensor(velocities, velocity_estimate_1, time_step/2.0, n);

    double *state_3_positions = sum_scalar_mult_tensor(positions, position_estimate_2, time_step, n);
    double *velocity_estimate_3 = calculate_acceleration(masses, state_3_positions, n);
    delete[] state_3_positions;

    double *position_estimate_3 = sum_scalar_mult_tensor(velocities, velocity_estimate_2, time_step, n);

    double *new_positions = new double[n*3];
    double *new_velocities = new double[n*3];
    double factor = time_step/6.0;
    for(int i = 0; i<n*3; i++){
    	new_velocities[i] = (velocity_estimate_0[i] + velocity_estimate_1[i]*2.0 + velocity_estimate_2[i]*2.0 + velocity_estimate_3[i]) * factor;
    	new_positions[i] = (position_estimate_0[i] + position_estimate_1[i]*2.0 + position_estimate_2[i]*2.0 + position_estimate_3[i]) * factor;
	}
	delete[] velocity_estimate_0;
	delete[] velocity_estimate_1;
	delete[] velocity_estimate_2;
	delete[] velocity_estimate_3;
	delete[] position_estimate_1;
	delete[] position_estimate_2;
	delete[] position_estimate_3;
    double **pair = new double*[2];
    pair[0] = new_positions;
    pair[1] = new_velocities;
    return pair;
}

// Calculate the maximum relative difference between two arrays of vectors
double max_error(double *current_position_change, double *previous_position_change, int n){
	double maximum_error = 0;
	for(int i = 0; i<n; i++){
		double numerator = 0;
		double denominator = 0;
		for(int k = 0; k<3; k++){
			numerator += pow(current_position_change[i*3+k]-previous_position_change[i*3+k], 2);
			denominator += pow(current_position_change[i*3+k], 2);
		}
		maximum_error = max(numerator/denominator, maximum_error);
	}
	return pow(maximum_error, 0.5);
}

// Calculate the evolution of the system over a time step time_step.
// Continue calculating to better and better approximation until
// max_error function returns 1e-5.
SimStep update_particles_recursive(double *masses, double *positions, double *velocities, int n, double time_step, double *previous_position_change, int iteration_limit, double time){
	SimStep ret;
	double **state_1 = runge_kutta(masses, positions, velocities, n, time_step / 2.0);
	double *state_1_positions = sum_tensors(positions, state_1[0], n);
	double *state_1_velocities = sum_tensors(velocities, state_1[1], n);
	double **state_2 = runge_kutta(masses, state_1_positions, state_1_velocities, n, time_step / 2.0);
	double *total_position_change = sum_tensors(state_1[0], state_2[0], n);
	if(iteration_limit>0 && max_error(total_position_change, previous_position_change, n) > 1e-5){
		delete[] state_1_positions;
		delete[] state_1_velocities;
    	SimStep recursed_state_1 = update_particles_recursive(masses, positions, velocities, n, time_step / 2, state_1[0], iteration_limit-1, time);
    	for(int i = 0; i<2; i++){
    		delete[] state_1[i];
    		delete[] state_2[i];
    	}
    	delete[] state_1;
    	delete[] state_2;
    	double *state_2_positions = sum_tensors(positions, total_position_change, n);
    	double *previous_positions_2 = sum_scalar_mult_tensor(state_2_positions, recursed_state_1.positions_array[recursed_state_1.depth-1], -1, n);
    	delete[] state_2_positions;
    	delete[] total_position_change;
    	SimStep recursed_state_2 = update_particles_recursive(masses, recursed_state_1.positions_array[recursed_state_1.depth-1], recursed_state_1.velocities_array[recursed_state_1.depth-1], n, time_step / 2, previous_positions_2, iteration_limit-1, recursed_state_1.times_array[recursed_state_1.depth-1]);
    	delete[] previous_positions_2;
    	ret = recursed_state_1+recursed_state_2;
    	return ret;
	}else{
		delete total_position_change;
		double *state_2_positions = sum_tensors(state_1_positions, state_2[0], n);
		double *state_2_velocities = sum_tensors(state_1_velocities, state_2[1], n);
    	ret = SimStep(state_1_positions, state_2_positions, state_1_velocities, state_2_velocities, time+time_step/2.0, time+time_step, n);
    	for(int i = 0; i<2; i++){
    		delete[] state_1[i];
    		delete[] state_2[i];
    	}
    	delete[] state_1;
    	delete[] state_2;
    	return ret;
	}
}

// Calculate the evolution of the system over a time step time_step.
SimStep update_particles(double *masses, double *positions, double *velocities, int n, double time_step){
    double **state_1 = runge_kutta(masses, positions, velocities, n, time_step);
    SimStep ret = SimStep(positions, velocities, 0, n) + update_particles_recursive(masses, positions, velocities, n, time_step, state_1[0], 30, 0);
    delete[] state_1[0];
    delete[] state_1[1];
    delete[] state_1;
    return ret;
}

// Convert python list type to C array type
double *convert_to_tensor(py::list &array, int n){
	double *tensor = new double[n*3];
	for(int i = 0; i<n*3; i++){
		tensor[i] = array[i].cast<double>();
	}
	return tensor;
}

// Convert C array type to python list type
void convert_to_list(py::list &array, double *tensor, int n){
	for(int i = 0; i<n*3; i++){
		array[i] = tensor[i];
	}
}

// Class exposed to python code that wraps the other functions
// in this file for easy interfacing.  Instantiate a new Simulator
// object with the desired start configuration and time step data.
// call step_forward() to update the system internally by time step.
// call get_positions() and get_velocities() to get the updated state.
class simulator{
public:
	double *masses;
	py::list positions;
	py::list velocities;
	double *position_tensor;
	double *velocity_tensor;
	double time_step;
	SimStep data;
	double last_time;
	double time;
	int n;

	simulator(py::list &masses, py::list &positions, py::list &velocities, double time_step, int n){
		this->n = n;
		this->masses = new double[n];
		for(int i = 0; i<n; i++){
			this->masses[i] = masses[i].cast<double>();
		}
		this->positions = positions;
		this->velocities = velocities;
		position_tensor = convert_to_tensor(positions, n);
		velocity_tensor = convert_to_tensor(velocities, n);
		this->time_step = time_step;
		this->last_time = 0;
		this->time = 0;
		this->data = SimStep();
	}

	~simulator(){
		delete[] masses;
	}

	void step_forward(){
		time += time_step;
		while(time > last_time){
			data.obliterate();
            data = update_particles(masses, copy_tensor(position_tensor, n), copy_tensor(velocity_tensor, n), n, time_step * 16);
            last_time += time_step * 16;
        }
        double relative_time = time - last_time + time_step * 16;
        data.update(relative_time, position_tensor, velocity_tensor, n);
        convert_to_list(positions, position_tensor, n);
        convert_to_list(velocities, velocity_tensor, n);
	}

	py::list get_positions(){
		return positions;
	}

	py::list get_velocities(){
		return velocities;
	}

	void print_state(){
		cout << "State at time: " << time << endl;
		for(int i = 0; i<n; i++){
			cout << i << ": ";
			for(int j = 0; j<3; j++){
				cout << position_tensor[i*3+j] << ", ";
			}
			cout << endl;
		}
	}
};

// Code to expose Simulator class to python using pybind11
PYBIND11_MODULE(Simulator, m) {
    m.doc() = "pybind11 simulator plugin"; // optional module docstring
    py::class_<simulator>(m, "simulator")
        .def(py::init<py::list &, py::list &, py::list &, double, int>())
        .def("step_forward", &simulator::step_forward)
        .def("get_positions", &simulator::get_positions)
        .def("get_velocities", &simulator::get_velocities);
}








