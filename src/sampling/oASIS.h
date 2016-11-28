#ifndef SPARSE_KERNEL_SAMPLING_OASIS_H
#define SPARSE_KERNEL_SAMPLING_OASIS_H

#include <data/Data.hpp>
#include <utility/RuntimeMonitor.hpp>

class oASIS {
public:
    oASIS(const Data *data, const uint64_t init_cols = 10, const uint64_t max_cols = 200, const float err_tolerance = 0.0f);
    ~oASIS(void);
    float GetError(const Data *data) const;
    float GetRuntime(void) const {
        return runtime_.get().count();
    }
private:
    Eigen::MatrixXf Winv_max_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ctransp_max_;
    uint64_t k_;
    RuntimeMonitor runtime_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_OASIS_H */
