#ifndef SPARSE_KERNEL_SAMPLING_MNIST_H
#define SPARSE_KERNEL_SAMPLING_MNIST_H

#include "Data.hpp"
#include <fstream>

class MNIST : public Data {
public:
    MNIST(int argc, char **argv);
    virtual ~MNIST(void);
    virtual uint64_t num_items (void) {
        return num_items_;
    }
protected:
    virtual float kernel_distance (uint64_t i, uint64_t j);
private:
    float two_sigma_squared_;
    typedef struct header {
        uint32_t magic;
        uint32_t num_items;
        uint32_t rows;
        uint32_t columns;
    } header_t;
    std::ifstream train_images_;
    uint32_t num_items_;
    uint32_t rows_;
    uint32_t columns_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_MNIST_H */
