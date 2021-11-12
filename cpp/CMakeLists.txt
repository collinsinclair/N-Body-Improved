cmake_minimum_required(VERSION 3.20)
project(new_n_body)

set(CMAKE_CXX_STANDARD 14)

include_directories()

add_executable(new_n_body
        cpp/forces.cpp
        cpp/forces.h
#        cpp/tests.cpp
#        cpp/tests.h
#        cpp/run_tests.cpp
        cpp/simulate.cpp)
