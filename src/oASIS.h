#ifndef SPARSE_KERNEL_SAMPLING_OASIS_H
#define SPARSE_KERNEL_SAMPLING_OASIS_H

#include "Data.hpp"

class oASIS {
public:
    oASIS(Data *data, const uint64_t init_cols = 10, const uint64_t max_cols = 200, const float err_tolerance = 0.0f);
    ~oASIS(void);
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_OASIS_H */
