#ifndef SPARSE_KERNEL_SAMPLING_ABALONE_HPP
#define SPARSE_KERNEL_SAMPLING_ABALONE_HPP

#include "Data.hpp"
#include <vector>
#include <eigen3/Eigen/Eigen>

class Abalone : public Data {
public:
    Abalone (int argc, char **argv);
    virtual ~Abalone (void);
    virtual uint64_t num_items (void) const {
        return G_.cols();
    }
    virtual Eigen::VectorXf column (uint64_t i) const {
        return G_.col(i);
    }
    virtual Eigen::RowVectorXf diagonal (void) const {
        return G_.diagonal();
    }
    virtual const Eigen::MatrixXf &G(void) const {
        return G_;
    }
private:
    Eigen::MatrixXf G_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_ABALONE_HPP */
