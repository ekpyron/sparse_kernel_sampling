#ifndef SPARSE_KERNEL_SAMPLING_DATA_H
#define SPARSE_KERNEL_SAMPLING_DATA_H

#include <cstdint>
#include <eigen3/Eigen/Eigen>
#include <numeric>

class Data {
public:
    Data (float jitter = std::numeric_limits<float>::epsilon()) : jitter_(jitter) {
    }
    virtual ~Data(void) {}
    virtual uint64_t num_items (void) = 0;
    virtual float distance (uint64_t i, uint64_t j) {
        if (i == j) {
            return kernel_distance(i,j) + jitter_;
        } else {
            return kernel_distance(i,j);
        }
    }
    virtual const Eigen::MatrixXf &G(void) const {
        static Eigen::MatrixXf G_;
        return G_;
    }
protected:
    virtual float kernel_distance (uint64_t i, uint64_t j) = 0;
private:
    float jitter_;
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_DATA_H */
