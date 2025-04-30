cmake_minimum_required(VERSION 3.25)

project(Simulator LANGUAGES CXX CUDA)
set(PYBIND11_CPP_STANDARD -std=c++11)

add_subdirectory(pybind11)

add_library(Simulatorcu STATIC
    simulator.cu
)

set_target_properties(Simulatorcu PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    # CUDA_SEPARABLE_COMPILATION ON
)


add_library(Simulator MODULE
    simulator.cpp
)

set_target_properties(Simulator PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(Simulator PRIVATE Simulatorcu)
target_link_libraries(Simulator PRIVATE pybind11::module)
