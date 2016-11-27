find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

set(SPARSE_KERNEL_SAMPLING_SOURCES main.cpp Data.hpp MNIST_mmap.cpp MNIST_mmap.hpp oASIS.cpp oASIS.h
                                   Abalone.cpp Abalone.hpp TwoMoons.cpp TwoMoons.hpp Nystrom.cpp Nystrom.hpp)

add_executable(sparse_kernel_sampling ${SPARSE_KERNEL_SAMPLING_SOURCES})
