cmake_minimum_required(VERSION 3.21)
project(ocl2)

set(OpenCL_INCLUDE_DIR "C:\\Users\\fedor\\GPU\\ocl2\\OpenCL-SDK\\external\\OpenCL-Headers")
set(OpenCL_LIBRARY "C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries\\OpenCL.lib")

find_package(OpenCL REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_definitions(-DLOCAL)
add_executable(ocl2 main.cpp)
target_link_libraries(ocl2 OpenCL::OpenCL)