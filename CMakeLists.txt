cmake_minimum_required(VERSION 3.18)
project(fisher)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c14")

find_package(CUDA REQUIRED)

foreach(ComputeCapability 60 61 70 72 75 86 90)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_${ComputeCapability},code=sm_${ComputeCapability})
endforeach()

if (MSVC)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;$<$<CONFIG:Debug>:/MDd>$<$<CONFIG:Release>:/MD>)
endif()

include_directories(${CMAKE_SOURCE_DIR}/lib/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
# link_directories(${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(src)