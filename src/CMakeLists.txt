
find_package(CUDAToolkit REQUIRED)
message(STATUS "CMake is using CUDA Toolkit version: ${CUDAToolkit_VERSION}")
include_directories(${CMAKE_SOURCE_DIR}/lib/include/culbfgsb)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/lib/include/utils)  
include_directories(${CMAKE_SOURCE_DIR}/cnpy)
# show all SRC
file(GLOB SRC "./*.cpp" "./*.cu")
file(GLOB HDR "./*.h" "./*.hpp")
message(SRC: ${SRC})
cuda_add_executable(fisher
  ${HDR}
  ${SRC}
)
target_compile_definitions(fisher PRIVATE
  FILE_IO_DIR="${CMAKE_SOURCE_DIR}/file_dir")
target_link_libraries(fisher 
    ${CMAKE_SOURCE_DIR}/lib/libcuLBFGSB.so 
    CUDA::cublas
)

target_link_libraries(fisher 
    ${CMAKE_SOURCE_DIR}/cnpy/build/libcnpy.so
)
# # Specify the minimum CMake version

# cmake_minimum_required(VERSION 3.18)
# set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)


# # Enable CUDA and C++ as project languages

# project(MyFisherProject LANGUAGES CXX CUDA)



# # Find the CUDA toolkit and its components

# find_package(CUDAToolkit REQUIRED)



# # Gather source and header files

# file(GLOB SRC "src/*.cpp" "src/*.cu")

# file(GLOB HDR "src/*.h")



# # Add include directories

# include_directories(${CMAKE_SOURCE_DIR}/lib/include/culbfgsb)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})



# # Create the executable from sources (replaces cuda_add_executable)

# add_executable(fisher ${HDR} ${SRC})



# # Link libraries using the modern keyword signature

# target_link_libraries(fisher 
#     ${CMAKE_SOURCE_DIR}/lib/libcuLBFGSB.so 
#     CUDA::cublas
# )