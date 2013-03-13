cmake_minimum_required (VERSION 2.6)
project (TOMOGRAPHY)
# The version number.
set (TOMOGRAPHY_VERSION_MAJOR 1)
set (TOMOGRAPHY_VERSION_MINOR 0)

#set(PLATFORM 64)
 
set(CMAKE_CXX_FLAGS "-g -Wall -O3 -funroll-loops -msse3")

#turn this on for shale
#set(BOOST_ROOT /home/tdesell/boost-build)
#set(CMAKE_INCLUDE_PATH /home/tdesell/boost-build/ ${CMAKE_INCLUDE_PATH})
#set(CMAKE_LIBRARY_PATH /home/tdesell/boost-build/ ${CMAKE_LIBRARY_PATH} /opt/local/lib)


set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/tao/undvc_common/cmake/Modules/")

set(BOOST_LIB_PREFIX "lib")
set(Boost_LIB_PREFIX "lib")
set(BOOST_COMPILER "-vc100")
set(Boost_COMPILER "-vc100")
set(Boost_ADDITIONAL_VERSIONS "1.47" "1.47.0")

find_package(Boost REQUIRED system filesystem)
find_package(MySQL)

# packages                                                                                                                                                                                                 
find_package(CUDA)

#The following two lines will use mvapich2 on hodor
#set (MPI_COMPILER "/opt/mvapich2-x/gnu/bin/mpic++")
#set (MPI_INCLUDE_PATH "/opt/mvapich2-x/gnu/include")

find_package(MPI)

# nvcc flags                                                                                                                                                                                               
add_subdirectory(tao)

#include_directories (${PROJECT_SOURCE_DIR}/tao/evolutionary_algorithms)
#include_directories (${PROJECT_SOURCE_DIR}/tao/synchronous_algorithms)
#include_directories (${PROJECT_SOURCE_DIR}/tao/undvc_common)
include_directories (${PROJECT_SOURCE_DIR}/tao)
include_directories (${Boost_INCLUDE_DIR})

include_directories (${MPI_INCLUDE_PATH})
add_executable(fdtd_cpu test FDTD_common FDTD_CPU)
target_link_libraries(fdtd_cpu undvc_common asynchronous_algorithms synchronous_algorithms tao_util mpi_algorithms)

#if (CUDA_FOUND) 
#use this for shale
#cuda_add_executable(fdtd_gpu test FDTD_GPU.cu OPTIONS --compiler-bindir mpicxx)
#    cuda_add_executable(fdtd_gpu test oldFDTD_GPU.cu OPTIONS -DCUDA)
#    target_link_libraries(fdtd_gpu undvc_common asynchronous_algorithms synchronous_algorithms tao_util mpi_algorithms)
#    set_target_properties(fdtd_gpu PROPERTIES COMPILE_DEFINITIONS CUDA)
#endif (CUDA_FOUND)

if (CUDA_FOUND) 
#use this for shale
#cuda_add_executable(fdtd_gpu test FDTD_GPU.cu FDTD_common FDTD_CPU OPTIONS --compiler-bindir mpicxx)
    cuda_add_executable(fdtd_gpu_new test FDTD_GPU.cu FDTD_common FDTD_CPU OPTIONS -DCUDA)
    target_link_libraries(fdtd_gpu_new undvc_common asynchronous_algorithms synchronous_algorithms tao_util mpi_algorithms)
    set_target_properties(fdtd_gpu_new PROPERTIES COMPILE_DEFINITIONS CUDA)
endif (CUDA_FOUND)



#if (MYSQL_FOUND)
#    include_directories(${MYSQL_INCLUDE_DIR})

#    add_executable(SecondStageDB second_stage_db)
#    target_link_libraries(SecondStageDB government_problem undvc_common asynchronous_algorithms synchronous_algorithms tao_util db_asynchronous_algorithms ${MYSQL_LIBRARIES})
#endif (MYSQL_FOUND)
