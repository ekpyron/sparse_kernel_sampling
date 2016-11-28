#ifndef SPARSE_KERNEL_SAMPLING_NYSTROM_HPP
#define SPARSE_KERNEL_SAMPLING_NYSTROM_HPP

#include "data/Data.hpp"
#include <eigen3/Eigen/Eigen>

class Nystrom {
public:
    Nystrom(const Data *data, const uint64_t k);
    ~Nystrom(void);
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &Ctransp(void) const {
        return Ctransp_;
    }
    const Eigen::MatrixXf &Winv(void) const {
        return Winv_;
    }
    const std::vector<uint64_t> &Lambda(void) const {
        return Lambda_;
    }
    void CheckResult (const Data *data);
private:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ctransp_;
    Eigen::MatrixXf Winv_;
    std::vector<uint64_t> Lambda_;
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_NYSTROM_HPP */
