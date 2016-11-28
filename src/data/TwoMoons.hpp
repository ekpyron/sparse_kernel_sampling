#ifndef SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP
#define SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP

#include "Data.hpp"
#include <vector>
#include <eigen3/Eigen/Eigen>

class TwoMoons : public Data {
public:
    TwoMoons(int argc, char **argv);
    virtual ~TwoMoons(void);
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


#endif /* !defined SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP */
