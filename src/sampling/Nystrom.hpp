#ifndef SPARSE_KERNEL_SAMPLING_NYSTROM_HPP
#define SPARSE_KERNEL_SAMPLING_NYSTROM_HPP

#include <data/Data.hpp>
#include <utility/RuntimeMonitor.hpp>
#include <eigen3/Eigen/Eigen>
#include <memory>

template<typename float_type = float>
class Nystrom {
public:
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixType;
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, 1, Eigen::ColMajor> VectorType;
    typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorType;

    Nystrom(const Data<float_type> *data, const uint64_t k, const std::shared_ptr<RuntimeMonitor> &runtime = std::make_shared<RuntimeMonitor>());
    ~Nystrom(void);
    const MatrixType &Winv(void) const {
        return Winv_;
    }
    const RowMatrixType &Ctransp(void) const {
        return Ctransp_;
    }
    const std::vector<uint64_t> &Lambda(void) const {
        return Lambda_;
    }
    float_type GetError (const Data<float_type> *data);
    float_type GetRuntime (void) {
        return runtime_->get().count();
    }
    uint64_t k(void) const {
        return k_;
    }
private:
    RowMatrixType Ctransp_;
    MatrixType Winv_;
    uint64_t k_;
    std::vector<uint64_t> Lambda_;
    std::shared_ptr<RuntimeMonitor> runtime_;
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_NYSTROM_HPP */
