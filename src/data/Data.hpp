#ifndef SPARSE_KERNEL_SAMPLING_DATA_H
#define SPARSE_KERNEL_SAMPLING_DATA_H

#include <cstdint>
#include <Eigen/Eigen>

template<typename float_type>
class Data {
public:
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, 1, Eigen::ColMajor> VectorType;
    typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorType;
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixType;

    virtual ~Data(void) {}
    virtual uint64_t num_items (void) const = 0;
    virtual VectorType column (uint64_t i) const = 0;
    virtual RowVectorType diagonal (void) const = 0;
    virtual const MatrixType &G(void) const {
        static MatrixType G_;
        return G_;
    }
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_DATA_H */
