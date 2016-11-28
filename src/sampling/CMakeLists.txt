set(SPARSE_KERNEL_SAMPLING_SAMPLING_SOURCES oASIS.cpp oASIS.h Nystrom.cpp Nystrom.hpp)

add_library(sparse_kernel_sampling_sampling STATIC ${SPARSE_KERNEL_SAMPLING_SAMPLING_SOURCES})
