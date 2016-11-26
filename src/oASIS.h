#ifndef SPARSE_KERNEL_SAMPLING_OASIS_H
#define SPARSE_KERNEL_SAMPLING_OASIS_H

#include "Data.h"

class oASIS {
public:
    oASIS(Data *data);
    ~oASIS(void);
private:
    Data *data_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_OASIS_H */
