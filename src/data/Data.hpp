#ifndef SPARSE_KERNEL_SAMPLING_DATA_H
#define SPARSE_KERNEL_SAMPLING_DATA_H

#include <cstdint>
#include <eigen3/Eigen/Eigen>
#include <numeric>

class Data {
public:
    virtual ~Data(void) {}
    virtual uint64_t num_items (void) const = 0;
    virtual Eigen::VectorXf column (uint64_t i) const = 0;
    virtual Eigen::RowVectorXf diagonal (void) const = 0;
    virtual const Eigen::MatrixXf &G(void) const {
        static Eigen::MatrixXf G_;
        return G_;
    }
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_DATA_H */
