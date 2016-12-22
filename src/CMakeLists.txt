find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPFR)

option(DISABLE_MPFR "Disable MPFR even if found." OFF)
if(MPFR_FOUND AND NOT DISABLE_MPFR)
    add_definitions(-DUSE_MPFR)
endif()

if (NOT ("${EIGEN3_VERSION_MAJOR}.${EIGEN3_VERSION_MINOR}" VERSION_LESS "3.3"))
    add_definitions(-DUSE_BDCSVD)
endif()

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${MPFR_INCLUDES})

add_subdirectory(data)
add_subdirectory(sampling)
add_subdirectory(MDS)
add_subdirectory(utility)

set(SPARSE_KERNEL_SAMPLING_SOURCES main.cpp)

add_executable(sparse_kernel_sampling ${SPARSE_KERNEL_SAMPLING_SOURCES})

target_link_libraries(sparse_kernel_sampling sparse_kernel_sampling_data sparse_kernel_sampling_sampling
        sparse_kernel_sampling_utility sparse_kernel_sampling_mds ${MPFR_LIBRARIES})
