#ifndef SPARSE_KERNEL_SAMPLING_NYSTROM_HPP
#define SPARSE_KERNEL_SAMPLING_NYSTROM_HPP

#include "Data.hpp"
#include <eigen3/Eigen/Eigen>

class Nystrom {
public:
    Nystrom(Data *data, const uint64_t k);
    ~Nystrom(void);
    const Eigen::MatrixXf &C(void) const {
        return C_;
    }
    const Eigen::MatrixXf &Winv(void) const {
        return Winv_;
    }
private:
    Eigen::MatrixXf C_;
    Eigen::MatrixXf Winv_;
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_NYSTROM_HPP */
