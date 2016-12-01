#ifndef SPARSE_KERNEL_SAMPLING_PSEUDOINV_HPP
#define SPARSE_KERNEL_SAMPLING_PSEUDOINV_HPP

#include <Eigen/Eigen>
#include <memory>

class RuntimeMonitor;

template<typename float_type>
struct PseudoInverse {
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixType;
    static MatrixType compute(MatrixType const& M, const std::shared_ptr<RuntimeMonitor> &runtime);
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_PSEUDOINV_HPP */