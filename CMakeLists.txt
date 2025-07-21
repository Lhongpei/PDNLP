cmake_minimum_required(VERSION 3.18)
project(fisher)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c14")

find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/lib/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
# link_directories(${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(src)