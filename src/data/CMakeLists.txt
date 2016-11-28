set(SPARSE_KERNEL_SAMPLING_DATA_SOURCES Data.hpp MNIST.cpp MNIST.hpp
        Abalone.cpp Abalone.hpp TwoMoons.cpp TwoMoons.hpp)

add_library(sparse_kernel_sampling_data STATIC ${SPARSE_KERNEL_SAMPLING_DATA_SOURCES})