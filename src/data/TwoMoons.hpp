#ifndef SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP
#define SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP

#include "Data.hpp"
#include <vector>
#include <eigen3/Eigen/Eigen>

template<typename float_type>
class TwoMoons : public Data<float_type> {
public:
    typedef typename Data<float_type>::VectorType VectorType;
    typedef typename Data<float_type>::RowVectorType RowVectorType;
    typedef typename Data<float_type>::MatrixType MatrixType;

    TwoMoons(void);
    virtual ~TwoMoons(void);
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


#endif /* !defined SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP */
