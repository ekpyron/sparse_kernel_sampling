#ifndef SPARSE_KERNEL_SAMPLING_DATA_H
#define SPARSE_KERNEL_SAMPLING_DATA_H

#include <cstdint>

class Data {
public:
    virtual ~Data(void) {}
    virtual uint64_t num_items (void) = 0;
    virtual float distance (uint64_t i, uint64_t j) = 0;
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_DATA_H */
