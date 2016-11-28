#ifndef SPARSE_KERNEL_SAMPLING_MNIST_MMAP_H
#define SPARSE_KERNEL_SAMPLING_MNIST_MMAP_H

#include "Data.hpp"
#include <cstdint>

template<typename float_type>
class MNIST : public Data<float_type> {
public:
    typedef typename Data<float_type>::VectorType VectorType;
    typedef typename Data<float_type>::RowVectorType RowVectorType;
    typedef typename Data<float_type>::MatrixType MatrixType;

    MNIST(void);
    virtual ~MNIST(void);
    virtual uint64_t num_items (void) const {
        return num_items_;
    }
    virtual VectorType column (uint64_t i) const;
    virtual RowVectorType diagonal (void) const;
protected:
    virtual float_type distance (uint64_t i, uint64_t j) const;
    float_type two_sigma_squared_;
    typedef struct header {
        uint32_t magic;
        uint32_t num_items;
        uint32_t rows;
        uint32_t columns;
    } header_t;
    int fd;
    uint8_t *mem;
    uint32_t num_items_;
    uint32_t rows_;
    uint32_t columns_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_MNIST_MMAP_H */
