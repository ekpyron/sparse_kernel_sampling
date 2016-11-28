#ifndef SPARSE_KERNEL_SAMPLING_NYSTROM_HPP
#define SPARSE_KERNEL_SAMPLING_NYSTROM_HPP

#include <data/Data.hpp>
#include <utility/RuntimeMonitor.hpp>
#include <eigen3/Eigen/Eigen>
#include <memory>

class Nystrom {
public:
    Nystrom(const Data *data, const uint64_t k, const std::shared_ptr<RuntimeMonitor> &runtime = std::make_shared<RuntimeMonitor>());
    ~Nystrom(void);
    const Eigen::MatrixXf &Winv(void) const {
        return Winv_;
    }
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &Ctransp(void) const {
        return Ctransp_;
    }
    const std::vector<uint64_t> &Lambda(void) const {
        return Lambda_;
    }
    float GetError (const Data *data);
    float GetRuntime (void) {
        return runtime_->get().count();
    }
private:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ctransp_;
    Eigen::MatrixXf Winv_;
    std::vector<uint64_t> Lambda_;
    std::shared_ptr<RuntimeMonitor> runtime_;
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_NYSTROM_HPP */
