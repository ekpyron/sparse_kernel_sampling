find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(data)

set(SPARSE_KERNEL_SAMPLING_SOURCES main.cpp oASIS.cpp oASIS.h Nystrom.cpp Nystrom.hpp RuntimeMonitor.cpp RuntimeMonitor.hpp)

add_executable(sparse_kernel_sampling ${SPARSE_KERNEL_SAMPLING_SOURCES})

target_link_libraries(sparse_kernel_sampling sparse_kernel_sampling_data)