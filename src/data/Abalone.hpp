#ifndef SPARSE_KERNEL_SAMPLING_ABALONE_HPP
#define SPARSE_KERNEL_SAMPLING_ABALONE_HPP

#include "Data.hpp"
#include <vector>
#include <Eigen/Eigen>

template<typename float_type>
class Abalone : public Data<float_type> {
public:
    typedef typename Data<float_type>::VectorType VectorType;
    typedef typename Data<float_type>::RowVectorType RowVectorType;
    typedef typename Data<float_type>::MatrixType MatrixType;

    Abalone (void);
    virtual ~Abalone (void);
    virtual uint64_t num_items (void) const {
        return G_.cols();
    }
    virtual VectorType column (uint64_t i) const {
        return G_.col(i);
    }
    virtual RowVectorType diagonal (void) const {
        return G_.diagonal();
    }
    virtual const MatrixType &G(void) const {
        return G_;
    }
private:
    MatrixType G_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_ABALONE_HPP */
