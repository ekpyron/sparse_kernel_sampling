#ifndef SPARSE_KERNEL_SAMPLING_OASIS_H
#define SPARSE_KERNEL_SAMPLING_OASIS_H

#include <data/Data.hpp>
#include <utility/RuntimeMonitor.hpp>
#include <memory>

class oASIS {
public:
    oASIS(const Data *data, const std::shared_ptr<RuntimeMonitor> &runtime = std::make_shared<RuntimeMonitor>());
    ~oASIS(void);
    float GetError(const Data *data) const;
    float GetRuntime(void) const {
        return runtime_->get().count();
    }
private:
    Eigen::MatrixXf Winv_max_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ctransp_max_;
    uint64_t k_;
    std::shared_ptr<RuntimeMonitor> runtime_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_OASIS_H */
