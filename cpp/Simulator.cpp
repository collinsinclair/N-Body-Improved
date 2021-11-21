//#include <pybind11/pybind11.h>

//namespace py = pybind11;
#include <math.h>
#include <iostream>
#include <ctime>
#include <string>

using namespace std;

#include <pybind11/pybind11.h>

namespace py = pybind11;

double *sumTensors(double *tensor1, double *tensor2, int n){
	double *newtensor = new double[n*3];
	for(int i = 0; i<n*3; i++){
		newtensor[i] = tensor1[i] + tensor2[i];
	}
	return newtensor;
}

double *scalarMultTensor(double *tensor, double scalar, int n){
	double *newtensor = new double[n*3];
	for(int i = 0; i<n*3; i++){
		newtensor[i] = tensor[i] * scalar;
	}
	return newtensor;
}

double *sumScalarMultTensor(double *tensor, double *tensor2, double scalar, int n){
	double *newtensor = new double[n*3];
	for(int i = 0; i<n*3; i++){
		newtensor[i] = tensor[i] + tensor2[i] * scalar;
	}
	return newtensor;
}

double *copyArr(double *original, int n){
	double *na = new double[n*3];
	for(int i = 0; i<n*3; i++){
		na[i] = original[i];
	}
	return na;
}

class Timer{
public:
	double t;
	Timer(){
		t = clock();
	}
	void start(){
		t = clock();
	}
	void time(){
		cout << "Time since start: " << (clock()-t) << endl;
	}
	void time(string task){
		cout << "Time to " << task << ": " << (clock()-t) << endl;
	}
};

class SimStep{
public:
	int n;
	int depth;
	double **parray;
	double **varray;
	double *tarray;

	SimStep(){
		depth = 0;
	}

	~SimStep(){
		if(depth > 0){
			delete[] parray;
			delete[] varray;
			delete[] tarray;
			depth = 0;
		}
	}

	void obliterate(){
		if(depth > 0){
			for(int i = 0; i<depth; i++){
				delete[] parray[i];
				delete[] varray[i];
			}
			delete[] parray;
			delete[] varray;
			delete[] tarray;
			depth = 0;
		}
	}

	SimStep(const SimStep &other){
		n = other.n;
		depth = other.depth;
		if(depth > 0){
			parray = new double*[depth];
			varray = new double*[depth];
			tarray = new double[depth];
			for(int i = 0; i<depth; i++){
				parray[i] = other.parray[i];
				varray[i] = other.varray[i];
				tarray[i] = other.tarray[i];
			}
		}
	}

	SimStep(double **parray, double **varray, double *tarray, int depth, int n){
		this->parray = parray;
		this->varray = varray;
		this->tarray = tarray;
		this->depth = depth;
		this->n = n;
	}

	SimStep(double *p0, double *v0, double t0, int n){
		parray = new double*[1];
		parray[0] = p0;
		varray = new double*[1];
		varray[0] = v0;
		tarray = new double[1];
		tarray[0] = t0;
		depth = 1;
		this->n = n;
	}

	SimStep(double *p0, double *p1, double *v0, double *v1, double t0, double t1, int n){
		parray = new double*[2];
		parray[0] = p0;
		parray[1] = p1;
		varray = new double*[2];
		varray[0] = v0;
		varray[1] = v1;
		tarray = new double[2];
		tarray[0] = t0;
		tarray[1] = t1;
		depth = 2;
		this->n = n;
	}

	SimStep operator+(SimStep other){
		double **np = new double*[depth+other.depth];
		double **nv = new double*[depth+other.depth];
		double *nt = new double[depth+other.depth];
		for(int i = 0; i<depth; i++){
			np[i] = parray[i];
			nv[i] = varray[i];
			nt[i] = tarray[i];
		}
		for(int i = 0; i<other.depth; i++){
			np[i+depth] = other.parray[i];
			nv[i+depth] = other.varray[i];
			nt[i+depth] = other.tarray[i];
		}
		return SimStep(np, nv, nt, depth+other.depth, n);
	}

	void update(double t, double *positions, double *velocities, int n){
		for(int i = 0; i<depth - 1; i++){
			if(tarray[i] <= t && tarray[i+1] >= t){
				double frac = (t - tarray[i])/(tarray[i+1] - tarray[i]);
				for(int j = 0; j<n*3; j++){
					positions[j] = parray[i][j]*(1-frac) + parray[i+1][j]*(frac);
					velocities[j] = varray[i][j]*(1-frac) + varray[i+1][j]*(frac);
				}
				break;
			}
		}
	}

	SimStep cumSum(double *positions, double *velocities, double time){
		double **np = new double*[depth+1];
		double **nv = new double*[depth+1];
		double *nt = new double[depth+1];
		np[0] = positions;
		nv[0] = velocities;
		nt[0] = time;
		for(int i = 0; i<depth; i++){
			np[i+1] = sumTensors(np[i], parray[i], n);
			delete[] parray[i];
			nv[i+1] = sumTensors(nv[i], varray[i], n);
			delete[] varray[i];
			nt[i+1] = nt[i]+tarray[i];
		}
		delete[] parray;
		delete[] varray;
		delete[] tarray;
		parray = np;
		varray = nv;
		tarray = nt;
		depth++;
		return *this;
	}

	void printTimes(){
		cout << "T: ";
		for(int i = 0; i<depth; i++){
			cout << tarray[i] << ", ";
		}
		cout << endl;
	}

	void operator=(const SimStep &other){
		if(depth>0){
			delete[] parray;
			delete[] varray;
			delete[] tarray;
		}
		n = other.n;
		depth = other.depth;
		if(depth > 0){
			parray = new double*[depth];
			varray = new double*[depth];
			tarray = new double[depth];
			for(int i = 0; i<depth; i++){
				parray[i] = other.parray[i];
				varray[i] = other.varray[i];
				tarray[i] = other.tarray[i];
			}
		}
	}
};

double *calculate_acceleration(double *masses, double *positions, int n){
	double G = 6.67e-11;
    double *acc = new double[n*3];
    for(int i = 0; i<n*3; i++){
    	acc[i] = 0;
    }
    for(int i = 0; i<n; i++){
	    for(int j = i+1; j<n; j++){
        	double r = 0;
        	for(int k = 0; k<3; k++){
        		r += pow(positions[i*3+k]-positions[j*3+k], 2);
        	}
        	double magnitude = pow(r, -1.5);
        	for(int k = 0; k<3; k++){
        		double kmagnitude = magnitude*(positions[j*3+k] - positions[i*3+k]);
        		acc[i*3+k] += masses[j]*kmagnitude;
        		acc[j*3+k] -= masses[i]*kmagnitude;
        	}
    	}
    }
    for(int i = 0; i<n*3; i++){
    	acc[i] *= G;
    }
    return acc;
}

double **rungekutta(double *masses, double *positions, double *velocities, int n, double dt){
    double *kv0 = calculate_acceleration(masses, positions, n);

    double *kr0 = velocities;

    double *tp1 = sumScalarMultTensor(positions, kr0, dt/2.0, n);
    double *kv1 = calculate_acceleration(masses, tp1, n);
    delete[] tp1;

    double *kr1 = sumScalarMultTensor(velocities, kv0, dt/2.0, n);

    double *tp2 = sumScalarMultTensor(positions, kr1, dt/2.0, n);
    double *kv2 = calculate_acceleration(masses, tp2, n);
    delete[] tp2;

    double *kr2 = sumScalarMultTensor(velocities, kv1, dt/2.0, n);

    double *tp3 = sumScalarMultTensor(positions, kr2, dt, n);
    double *kv3 = calculate_acceleration(masses, tp3, n);
    delete[] tp3;

    double *kr3 = sumScalarMultTensor(velocities, kv2, dt, n);

    double *nr = new double[n*3];
    double *nv = new double[n*3];
    double factor = dt/6.0;
    for(int i = 0; i<n*3; i++){
    	nv[i] = (kv0[i] + kv1[i]*2.0 + kv2[i]*2.0 + kv3[i]) * factor;
    	nr[i] = (kr0[i] + kr1[i]*2.0 + kr2[i]*2.0 + kr3[i]) * factor;
	}
	delete[] kv0;
	delete[] kv1;
	delete[] kv2;
	delete[] kv3;
	delete[] kr1;
	delete[] kr2;
	delete[] kr3;
    double **pair = new double*[2];
    pair[0] = nr;
    pair[1] = nv;
    return pair;
}

double maxError(double *curr, double *prev, int n){
	double merr = 0;
	for(int i = 0; i<n; i++){
		double top = 0;
		double bot = 0;
		for(int k = 0; k<3; k++){
			top += pow(curr[i*3+k]-prev[i*3+k], 2);
			bot += pow(curr[i*3+k], 2);
		}
		merr = max(top/bot, merr);
	}
	return pow(merr, 0.5);
}

SimStep update_particles_recursive(double *masses, double *positions, double *velocities, int n, double dt, double *prev, int nmax, double time){
	SimStep ret;
	double **n1 = rungekutta(masses, positions, velocities, n, dt / 2.0);
	double *tp1 = sumTensors(positions, n1[0], n);
	double *tv1 = sumTensors(velocities, n1[1], n);
	double **n2 = rungekutta(masses, tp1, tv1, n, dt / 2.0);
	double *dp = sumTensors(n1[0], n2[0], n);
	if(nmax>0 && maxError(dp, prev, n) > 1e-3){
		delete[] tp1;
		delete[] tv1;
    	SimStep Nn1 = update_particles_recursive(masses, positions, velocities, n, dt / 2, n1[0], nmax-1, time);
    	for(int i = 0; i<2; i++){
    		delete[] n1[i];
    		delete[] n2[i];
    	}
    	delete[] n1;
    	delete[] n2;
    	double *ap = sumTensors(positions, dp, n);
    	double *tprev = sumScalarMultTensor(ap, Nn1.parray[Nn1.depth-1], -1, n);
    	delete[] ap;
    	delete[] dp;
    	SimStep Nn2 = update_particles_recursive(masses, Nn1.parray[Nn1.depth-1], Nn1.varray[Nn1.depth-1], n, dt / 2, tprev, nmax-1, Nn1.tarray[Nn1.depth-1]);
    	delete[] tprev;
    	ret = Nn1+Nn2;
    	return ret;
	}else{
		delete dp;
		double *tp2 = sumTensors(tp1, n2[0], n);
		double *tv2 = sumTensors(tv1, n2[1], n);
    	ret = SimStep(tp1, tp2, tv1, tv2, time+dt/2.0, time+dt, n);
    	for(int i = 0; i<2; i++){
    		delete[] n1[i];
    		delete[] n2[i];
    	}
    	delete[] n1;
    	delete[] n2;
    	return ret;
	}
}

SimStep update_particles(double *masses, double *positions, double *velocities, int n, double dt){
    double **na = rungekutta(masses, positions, velocities, n, dt);
    SimStep ret = SimStep(positions, velocities, 0, n) + update_particles_recursive(masses, positions, velocities, n, dt, na[0], 20, 0);
    delete[] na[0];
    delete[] na[1];
    delete[] na;
    return ret;
}

double *createArray(py::list &array, int n){
	double *arrayT = new double[n*3];
	for(int i = 0; i<n*3; i++){
		arrayT[i] = array[i].cast<double>();
	}
	return arrayT;
}

void convert(py::list &array, double *arrayT, int n){
	for(int i = 0; i<n*3; i++){
		array[i] = arrayT[i];
	}
}

class simulator{
public:
	double *masses;
	py::list positions;
	py::list velocities;
	double *positionsT;
	double *velocitiesT;
	double dt;
	SimStep data;
	double last_time;
	double time;
	int n;

	simulator(py::list &masses, py::list &positions, py::list &velocities, double dt, int n){
		this->n = n;
		this->masses = new double[n];
		for(int i = 0; i<n; i++){
			this->masses[i] = masses[i].cast<double>();
		}
		this->positions = positions;
		this->velocities = velocities;
		positionsT = createArray(positions, n);
		velocitiesT = createArray(velocities, n);
		this->dt = dt;
		this->last_time = 0;
		this->time = 0;
		this->data = SimStep();
	}

	~simulator(){
		delete[] masses;
	}

	void stepForward(){
		time += dt;
		while(time > last_time){
			data.obliterate();
            data = update_particles(masses, copyArr(positionsT, n), copyArr(velocitiesT, n), n, dt * 16);
            last_time += dt * 16;
        }
        double stime = time - last_time + dt * 16;
        data.update(stime, positionsT, velocitiesT, n);
        //printState();
        //data.printTimes();
        convert(positions, positionsT, n);
        convert(velocities, velocitiesT, n);
	}

	py::list getPositions(){
		return positions;
	}

	py::list getVelocities(){
		return velocities;
	}

	void printState(){
		cout << "State at time: " << time << endl;
		for(int i = 0; i<n; i++){
			cout << i << ": ";
			for(int j = 0; j<3; j++){
				cout << positionsT[i*3+j] << ", ";
			}
			cout << endl;
		}
	}
};

/*int main(){
	double *masses = new double[2];
	masses[0] = 1.989e30;
	masses[1] = 5.972e24;
	double G = 6.67e-11;
	double a = 1.496e11;
	double v_circular = pow(G*masses[0]/a, 0.5);
	double q = masses[1]/masses[0];
	Tensor<Vector> positions(2);
	positions[0] = Vector(-q*a, 0, 0);
	positions[1] = Vector((1-q)*a, 0, 0);
	Tensor<Vector> velocities(2);
	velocities[0] = Vector(0, -q*v_circular, 0);
	velocities[1] = Vector(0, (1-q)*v_circular, 0);
	double dt = 5*24*60*60;
	double *poss = new double[2*3];
	convert(poss, positions);
	double *vels = new double[2*3];
	convert(vels, velocities);
	simulator s(masses, poss, vels, dt, 2);
	s.printState();
	for(int i = 0; i<73; i++){
		s.stepForward();
		s.printState();
	}
	return 0;
}*/

PYBIND11_MODULE(Simulator, m) {
    m.doc() = "pybind11 simulator plugin"; // optional module docstring
    py::class_<simulator>(m, "simulator")
        .def(py::init<py::list &, py::list &, py::list &, double, int>())
        .def("stepForward", &simulator::stepForward)
        .def("getPositions", &simulator::getPositions)
        .def("getVelocities", &simulator::getVelocities);
}








