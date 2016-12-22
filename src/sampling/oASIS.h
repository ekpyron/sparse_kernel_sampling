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

    const MatrixType &W(void) const  {
        return W_;
    }
    const MatrixType &U(void) const {
        return U_;
    }
    RowMatrixType Ctransp(void) const {
        return Ctransp_max_.topRows(k_);
    }

    const std::vector<uint64_t> &Lambda(void) const {
        return Lambda_;
    }

private:
    MatrixType W_;
    MatrixType Winv_max_;
    RowMatrixType Ctransp_max_;
    uint64_t k_;
    std::vector<uint64_t> Lambda_;
    std::shared_ptr<RuntimeMonitor> runtime_;

    RowVectorType Sigma_;
    MatrixType U_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_OASIS_H */
