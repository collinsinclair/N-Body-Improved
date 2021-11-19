//#include <pybind11/pybind11.h>

//namespace py = pybind11;
#include <math.h>
#include <iostream>

using namespace std;

#include <pybind11/pybind11.h>

namespace py = pybind11;


class Vector{
public:
	double *array;
	int n;
	Vector(){
		n = 3;
		array = new double[3];
		for(int i = 0; i<3; i++){
			array[i] = 0;
		}
	}

	Vector(int n){
		this->n = n;
		array = new double[n];
		for(int i = 0; i<n; i++){
			array[i] = 0;
		}
	}

	Vector(double x, double y, double z){
		n = 3;
		array = new double[3];
		array[0] = x;
		array[1] = y;
		array[2] = z;
	}

	Vector(double *array, int n){
		this->n = n;
		this->array = new double[n];
		for(int i = 0; i<n; i++){
			this->array[i] = array[i];
		}
	}

	Vector(const Vector &other){
		n = other.n;
		array = new double[n];
		for(int i = 0; i<n; i++){
			array[i] = other.array[i];
		}
	}

	~Vector(){
		//cout << "Deleting Vector" << endl;
		if(n>0){
			//cout << n << endl;
			delete[] array;
		}
		//cout << "Vector Deleted" << endl;
	}

	Vector operator+(Vector other){
		if(this->n == other.n){
			double *na = new double[this->n];
			for(int i = 0; i<this->n; i++){
				na[i] = this->array[i] + other.array[i];
			}
			Vector ret(na, this->n);
			delete[] na;
			return ret;
		}
		Vector ret(0);
		return ret;
	}

	Vector operator-(Vector other){
		if(this->n == other.n){
			double *na = new double[n];
			for(int i = 0; i<this->n; i++){
				na[i] = this->array[i] - other.array[i];
			}
			Vector ret(na, this->n);
			delete[] na;
			return ret;
		}
		Vector ret(0);
		return ret;
	}

	double operator*(Vector other){
		if(this->n == other.n){
			double na = 0;
			for(int i = 0; i<n; i++){
				na += this->array[i] * other.array[i];
			}
			return na;
		}
		return 0;
	}

	Vector operator*(double other){
		double *na = new double[this->n];
		for(int i = 0; i<this->n; i++){
			na[i] = this->array[i] * other;
		}
		Vector ret(na, this->n);
		delete[] na;
		return ret;
	}

	Vector operator/(double other){
		double *na = new double[this->n];
		for(int i = 0; i<this->n; i++){
			na[i] = this->array[i] / other;
		}
		Vector ret(na, this->n);
		delete[] na;
		return ret;
	}

	double mag(){
		return pow((*this)*(*this), 0.5);
	}

	double &operator[](int i){
		return array[i];
	}

	void operator=(const Vector &other){
		n = other.n;
		array = new double[n];
		for(int i = 0; i<n; i++){
			array[i] = other.array[i];
		}
	}
};

template <typename T>
class Tensor{
public:
	int n;
	T *array;

	Tensor(){
		n = 0;
	}

	Tensor(const Tensor<T> &t){
		n = t.n;
		array = new T[n];
		for(int i = 0; i<n; i++){
			array[i] = t.array[i];
		}
	}

	Tensor(int n){
		this->n = n;
		this->array = new T[n];
	}

	Tensor(int n, int m){
		this->n = n;
		this->array = new Vector[n];
		for(int i = 0; i<n; i++){
			this->array[i] = Vector(m);
		}
	}

	Tensor(T *array, int n){
		this->n = n;
		this->array = new T[n];
		for(int i = 0; i<n; i++){
			this->array[i] = array[i];
		}
	}

	Tensor(double **array, int n){
		this->n = n;
		this->array = new T[n];
		for(int i = 0; i<n; i++){
			this->array[i] = Vector(array[i], 3);
		}
	}

	Tensor(py::list &array, int n, int m){
		this->n = n;
		this->array = new T[n];
		for(int i = 0; i<n; i++){
			double *na = new double[3];
			for(int j = 0; j<m; j++){
				na[j] = array[m*i+j].cast<double>();
			}
			this->array[i] = Vector(na, m);
		}
	}

	~Tensor(){
		//cout << "Deleting Tensor" << endl;
		if(n>0){
			//cout << n << endl;
			delete[] array;
		}
		//cout << "Tensor Deleted" << endl;
	}

	template <typename U>
	Tensor operator+(Tensor<U> other){
		if(this->n == other.n){
			T *na = new T[n];
			for(int i = 0; i<n; i++){
				na[i] = this->array[i] + other.array[i];
			}
			Tensor<T> ret(na, n);
			delete[] na;
			return ret;
		}
		return Tensor<T>();
	}

	template <typename U>
	Tensor operator-(Tensor<U> other){
		if(this->n == other.n){
			T *na = new T[n];
			for(int i = 0; i<n; i++){
				na[i] = this->array[i] - other.array[i];
			}
			Tensor<T> ret(na, n);
			delete[] na;
			return ret;
		}
		return Tensor<T>();
	}

	template <typename U>
	Tensor operator*(Tensor<U> other){
		if(this->n == other.n){
			T *na = new T[n];
			for(int i = 0; i<n; i++){
				na[i] = this->array[i] * other.array[i];
			}
			Tensor<T> ret(na, n);
			delete[] na;
			return ret;
		}
	}

	Tensor operator*(double other){
		T *na = new T[n];
		for(int i = 0; i<n; i++){
			na[i] = this->array[i] * other;
		}
		Tensor<T> ret(na, n);
		delete[] na;
		return ret;
	}

	template <typename U>
	Tensor operator/(Tensor<U> other){
		if(this->n == other.n){
			T *na = new T[n];
			for(int i = 0; i<n; i++){
				na[i] = this->array[i] / other.array[i];
			}
			Tensor<T> ret(na, n);
			delete[] na;
			return ret;
		}
	}

	Tensor operator/(double other){
		T *na = new T[n];
		for(int i = 0; i<n; i++){
			na[i] = this->array[i] / other;
		}
		Tensor<T> ret(na, n);
		delete[] na;
		return ret;
	}

	T &operator[](int index){
		return array[index];
	}

	void operator=(const Tensor<T> &other){
		n = other.n;
		array = new T[n];
		for(int i = 0; i<n; i++){
			array[i] = other.array[i];
		}
	}
};

template <typename T>
class Splitter{
public:
	int n;
	T *array;

	Splitter(const Splitter<T> &other){
		n = other.n;
		array = new T[n];
		for(int i = 0; i<n; i++){
			array[i] = other.array[i];
		}
	}

	Splitter(){
		n = 0;
	}

	Splitter(T first){
		this->n = 1;
		this->array = new T[1];
		this->array[0] = first;
	}

	Splitter(T first, T second){
		n = 2;
		array = new T[2];
		array[0] = first;
		//cout << "First set" << endl;
		array[1] = second;
		//cout << "Second set" << endl;
	}

	Splitter(T *array, int n){
		this->n = n;
		this->array = new T[n];
		for(int i = 0; i<n; i++){
			this->array[i] = array[i];
		}
	}

	~Splitter(){
		//cout << "Deleting Splitter" << endl;
		if(n>0){
			//cout << n << endl;
			delete[] array;
		}
		//cout << "Splitter Deleted" << endl;
	}

	Splitter<T> replace(int i, Splitter<T> other){
		int nn = this->n+other.n-1;
		T *na = new T[nn];
		for(int j = 0; j<i; j++){
			na[j] = this->array[j];
		}
		for(int j = 0; j<other.n; j++){
			na[i+j] = other.array[j];
		}
		for(int j = i+1; j<this->n; j++){
			na[other.n+j-1] = this->array[j];
		}
		Splitter ret(na, nn);
		delete[] na;
		return ret;
	}

	Splitter<T> operator+(Splitter<T> other){
		int nn = this->n+other.n;
		T *na = new T[nn];
		for(int j = 0; j<this->n; j++){
			na[j] = this->array[j];
		}
		for(int j = 0; j<other.n; j++){
			na[this->n+j] = other.array[j];
		}
		Splitter ret(na, nn);
		delete[] na;
		return ret;
	}

	Splitter<T> operator+(T other){
		int nn = this->n+1;
		T *na = new T[nn];
		for(int j = 0; j<this->n; j++){
			na[j] = this->array[j];
		}
		na[this->n] = other;
		Splitter ret(na, nn);
		delete[] na;
		return ret;
	}

	T &operator[](int i){
		return array[i];
	}

	void operator=(const Splitter<T> &other){
		n = other.n;
		array = new T[n];
		for(int i = 0; i<n; i++){
			array[i] = other.array[i];
		}
	}
};

class SimStep{
public:
	int n;
	Splitter<Tensor<Vector> > parray;
	Splitter<Tensor<Vector> > varray;
	Splitter<double> tarray;

	SimStep(){
		n = 0;
		//parray = Splitter<Tensor<Vector> >();
		//varray = Splitter<Tensor<Vector> >();
		//tarray = Splitter<double>();
	}

	SimStep(Splitter<Tensor<Vector> > parray, Splitter<Tensor<Vector> > varray, Splitter<double> tarray){
		this->parray = parray;
		this->varray = varray;
		this->tarray = tarray;
		this->n = this->parray.n;
	}

	/*~SimStep(){
		cout << "Deleting SimStep" << endl;
		if(n>0){
			delete &parray;
			cout << "parray deleted" << endl;
			delete &varray;
			cout << "varray deleted" << endl;
			delete &tarray;
		}
	}*/

	SimStep operator+(SimStep other){
		return SimStep(parray + other.parray, varray + other.varray, tarray + other.tarray);
	}

	Tensor<Tensor<Vector> > operator[](double t){
		Tensor<Tensor<Vector> > ret(2);
		for(int i = 0; i<n - 1; i++){
			if(tarray[i] <= t && tarray[i+1] >= t){
				double frac = (t - tarray[i])/(tarray[i+1] - tarray[i]);
				ret[0] = parray[i]*(1-frac) + parray[i+1]*(frac);
				ret[1] = varray[i]*(1-frac) + varray[i+1]*(frac);
				break;
			}
		}
		return ret;
	}

	SimStep cumSum(Tensor<Vector> positions, Tensor<Vector> velocities, double time){
		Splitter<Tensor<Vector> > prunner(positions);
		Splitter<Tensor<Vector> > vrunner(velocities);
		Splitter<double> trunner(time);
		for(int i = 0; i<this->n; i++){
			prunner = prunner + (prunner[i] + this->parray[i]);
			vrunner = vrunner + (vrunner[i] + this->varray[i]);
			trunner = trunner + (trunner[i] + this->tarray[i]);
		}
		return SimStep(prunner, vrunner, trunner);
	}

	void printTimes(){
		cout << "T: ";
		for(int i = 0; i<n; i++){
			cout << tarray[i] << ", ";
		}
		cout << endl;
	}
};

Tensor<Vector> calculate_acceleration(double *masses, Tensor<Vector> positions){
    int N = positions.n;
    double G = 6.67e-11;
    Tensor<Vector> acc(N, 3);
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            if(j != i){
            	double r = (positions[i]-positions[j]).mag();
            	double magnitude = G*masses[j]/pow(r, 3);
            	acc.array[i] = acc.array[i] + (positions[j] - positions[i])*(magnitude);
            }
    	}
    }
    return acc;
}

Tensor<Tensor<Vector> > rungekutta(double *masses, Tensor<Vector> positions, Tensor<Vector> velocities, double dt){
    Tensor<Vector> kv0 = calculate_acceleration(masses, positions);
    Tensor<Vector> kr0 = velocities;
    Tensor<Vector> kv1 = calculate_acceleration(masses, positions + kr0 * (dt / 2.0));
    Tensor<Vector> kr1 = velocities + kv0 * (dt / 2.0);
    Tensor<Vector> kv2 = calculate_acceleration(masses, positions + kr1 * (dt / 2.0));
    Tensor<Vector> kr2 = velocities + kv1 * (dt / 2.0);
    Tensor<Vector> kv3 = calculate_acceleration(masses, positions + kr2 * dt);
    Tensor<Vector> kr3 = velocities + kv2 * dt;
    Tensor<Vector> nv = (kv0 + kv1*2.0 + kv2*2.0 + kv3) * (dt/6.0);
    Tensor<Vector> nr = (kr0 + kr1*2.0 + kr2*2.0 + kr3) * (dt/6.0);
    Tensor<Tensor<Vector> > pair(2);
    pair[0] = nr;
    pair[1] = nv;
    return pair;
}

SimStep update_particles_recursive(double *masses, Tensor<Vector> positions, Tensor<Vector> velocities, double dt, Tensor<Vector> prev, int nmax, double time){
	SimStep ret;
	int n_particles = positions.n;
	Tensor<Tensor<Vector> > n1 = rungekutta(masses, positions, velocities, dt / 2);
	Tensor<Tensor<Vector> > n2 = rungekutta(masses, positions + n1[0], velocities + n1[1], dt / 2);
	//cout << "Second runges" << endl;
	double maxerror = 0;
	for(int i = 0; i<n_particles; i++){
		maxerror = max(maxerror, (n1[0][i] + n2[0][i] - prev[i]).mag()/(n1[0][i] + n2[0][i]).mag());
	}
	if(maxerror > 1e-2 and nmax>0){
    	SimStep Nn1 = update_particles_recursive(masses, positions, velocities, dt / 2, n1[0], nmax-1, time);
    	SimStep Nn2 = update_particles_recursive(masses, Nn1.parray[Nn1.n-1], Nn1.varray[Nn1.n-1], dt / 2, positions + n1[0] + n2[0] - Nn1.parray[Nn1.n-1], nmax-1, Nn1.tarray[Nn1.n-1]);
    	//cout << "MORE ACCURACY" << endl;
    	ret = Nn1+Nn2;
    	return ret;
	}else{
    	ret = SimStep(Splitter<Tensor<Vector> >(n1[0], n2[0]), Splitter<Tensor<Vector> >(n1[1], n2[1]), Splitter<double>(dt / 2, dt / 2)).cumSum(positions, velocities, time);
    	return ret;
	}
}

SimStep update_particles(double *masses, Tensor<Vector> positions, Tensor<Vector> velocities, double dt){
    Tensor<Tensor<Vector> > n = rungekutta(masses, positions, velocities, dt);
    //cout << "First Runge" << endl;
    return update_particles_recursive(masses, positions, velocities, dt, n[0], 20, 0);
}

void convert(py::list &array, Tensor<Vector> &arrayT){
	for(int i = 0; i<arrayT.n; i++){
		for(int j = 0; j<3; j++){
			array[i*3+j] = arrayT[i][j];
		}
	}
}

class simulator{
public:
	double *masses;
	py::list positions;
	py::list velocities;
	Tensor<Vector> positionsT;
	Tensor<Vector> velocitiesT;
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
		positionsT = Tensor<Vector>(positions, n, 3);
		velocitiesT = Tensor<Vector>(velocities, n, 3);
		this->dt = dt;
		this->last_time = 0;
		this->time = 0;
		this->data = SimStep();
	}

	void stepForward(){
		time += dt;
		while(time > last_time){
            data = update_particles(masses, positionsT, velocitiesT, dt * 16);
            //cout << data.n << endl;
            last_time += dt * 16;
        }
        double stime = time - last_time + dt * 16;
        Tensor<Tensor<Vector> > state = data[stime];
        //cout << "State x: " << state[0][0][0] << endl;
        positionsT = state[0];
        convert(positions, positionsT);
        velocitiesT = state[1];
        convert(velocities, positionsT);
	}

	py::list getPositions(){
		return positions;
	}

	py::list getVelocities(){
		return velocities;
	}

	void printState(){
		cout << "State at time: " << time << endl;
		int n = positionsT.n;
		for(int i = 0; i<n; i++){
			cout << i << ": ";
			for(int j = 0; j<3; j++){
				cout << positionsT[i][j] << ", ";
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








