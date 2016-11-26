find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

set(SPARSE_KERNEL_SAMPLING_SOURCES main.cpp Data.h MNIST.cpp MNIST.h oASIS.cpp oASIS.h)

add_executable(sparse_kernel_sampling ${SPARSE_KERNEL_SAMPLING_SOURCES})
