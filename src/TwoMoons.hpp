#ifndef SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP
#define SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP

#include "Data.hpp"
#include <vector>
#include <eigen3/Eigen/Eigen>

class TwoMoons : public Data {
public:
    TwoMoons(int argc, char **argv);
    virtual ~TwoMoons(void);
    virtual uint64_t num_items (void) {
        return G_.cols();
    }
    virtual const Eigen::MatrixXf &G(void) const {
        return G_;
    }
protected:
    virtual float kernel_distance (uint64_t i, uint64_t j) {
        return G_(i,j);
    }
private:
    Eigen::MatrixXf G_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_TWOMOONS_HPP */
