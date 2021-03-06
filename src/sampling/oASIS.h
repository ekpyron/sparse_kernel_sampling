#ifndef SPARSE_KERNEL_SAMPLING_OASIS_H
#define SPARSE_KERNEL_SAMPLING_OASIS_H

#include <data/Data.hpp>
#include <utility/RuntimeMonitor.hpp>
#include <memory>

template<typename float_type = float>
class oASIS {
public:
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixType;
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, 1, Eigen::ColMajor> VectorType;
    typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorType;

    oASIS(const Data<float_type> *data, const std::shared_ptr<RuntimeMonitor> &runtime = std::make_shared<RuntimeMonitor>());
    ~oASIS(void);
    float_type GetError(const Data<float_type> *data) const;
    float_type GetRuntime(void) const {
        return runtime_->get().count();
    }
    uint64_t k(void) const {
        return k_;
    }
private:
    MatrixType Winv_max_;
    RowMatrixType Ctransp_max_;
    uint64_t k_;
    std::shared_ptr<RuntimeMonitor> runtime_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_OASIS_H */
