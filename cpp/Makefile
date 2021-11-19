all: simulator

pythonpath = /opt/homebrew/opt/python@3.9/Frameworks/Python.framework/Versions/3.9/include/python3.9
pybind_includes = -I$(pythonpath)
suffix = .cpython-39-darwin.so

simulator:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(pybind_includes) Simulator.cpp -o Simulator$(suffix)