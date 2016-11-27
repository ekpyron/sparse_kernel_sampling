#ifndef SPARSE_KERNEL_SAMPLING_ABALONE_HPP
#define SPARSE_KERNEL_SAMPLING_ABALONE_HPP

#include "Data.hpp"
#include <vector>
#include <eigen3/Eigen/Eigen>

class Abalone : public Data {
public:
    Abalone (int argc, char **argv);
    virtual ~Abalone (void);
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


#endif /* !defined SPARSE_KERNEL_SAMPLING_ABALONE_HPP */
