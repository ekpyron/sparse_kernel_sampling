#ifndef SPARSE_KERNEL_SAMPLING_MDS_HPP
#define SPARSE_KERNEL_SAMPLING_MDS_HPP

#include <Eigen/Eigen>

template<typename float_type>
class MDS {
public:
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixType;
    typedef Eigen::Matrix<float_type, Eigen::Dynamic, 1, Eigen::ColMajor> VectorType;
    typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorType;

    MDS(const MatrixType &distances);
    ~MDS(void);

    const MatrixType &Lt (void) const {
        return Lt_;
    }
    const VectorType &avg (void) const {
        return avg_;
    }

private:
    MatrixType Lt_;
    VectorType avg_;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_MDS_HPP */
