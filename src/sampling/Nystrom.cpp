#include "Nystrom.hpp"
#include <Eigen/Eigen>
#include <random>
#include <iostream>
#ifdef USE_MPFR
#include <mpreal.h>
#endif
#include <utility/mymath.hpp>
#include <utility/Arguments.hpp>
#include <utility/PseudoInv.hpp>

template<typename float_type>
Nystrom<float_type>::Nystrom(const Data<float_type>* data, const uint64_t k, bool compute_svd,
  const std::shared_ptr<RuntimeMonitor> &runtime)
        : Ctransp_(k, data->num_items()), k_ (k), runtime_(runtime), have_svd_ (compute_svd) {
    uint64_t nitems = data->num_items();

    {
        RuntimeMonitorScope scope (*runtime_, "Choose ", k, " columns");
        std::random_device random_device;
        std::mt19937 rnd(random_device());
        for (uint64_t i = 0; i < k; i++) {
            uint64_t col = std::uniform_int_distribution<uint64_t> (0, nitems - k) (rnd) + i;
            Lambda_.push_back(col);
        }

    }


    {
        RuntimeMonitorScope scope (*runtime_, "Compute C_k^T");
        uint64_t j = 0;
        for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
            Ctransp_.row(j++) = data->column(*it);
        }
    }

    MatrixType W;
    {
        RuntimeMonitorScope scope (*runtime_, "Fetch W");
        W = MatrixType (k, k);
        uint64_t i = 0;
        for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
            W.row(i) = Ctransp_.col(*it);
            i++;
        }
    }

    if (compute_svd) {
        ScaleSVD(Sigma_, U_, W, Ctransp_, nitems, runtime_);
        Ctransp_.resize(0,0);
    } else {
        /*
#ifdef USE_BDCSVD
        Eigen::BDCSVD<MatrixType> SVD;
#else
        #warning "BDCSVD not available. Falling back to JacobiSVD"
    Eigen::JacobiSVD<MatrixType, Eigen::NoQRPreconditioner> SVD;
#endif
        {
            RuntimeMonitorScope scope (*runtime_, "Compute SVD of W");
            SVD = SVD.compute(W, Eigen::ComputeThinU);
        }

        int small_singular_values = 0;
        float_type cutoff = float_type(1e2)*std::numeric_limits<float_type>::epsilon();
        {
            RuntimeMonitorScope scope (*runtime_, "Compute W^{-1} (", SVD.nonzeroSingularValues(), ")");
            auto singValInv = SVD.singularValues();
            for (auto i = 0; i < singValInv.rows(); i++) {
                float_type &v = singValInv(i);
                if (my_abs(v)<=cutoff) {
                    small_singular_values++;
                    v = float_type(0.0);
                } else {
                    v = (float_type(1.0)/v);
                }
            }
            Winv_ = SVD.matrixU() * (singValInv.asDiagonal()) * SVD.matrixU().transpose();
        }
        if (small_singular_values && Arguments::get().verbose()) {
            std::cout << "  (" << small_singular_values << " singular values < " << cutoff << " were cut off)" << std::endl;
        }
         */
        Winv_ = PseudoInverse<float_type>::compute(W, runtime_);
    }
}

template<typename float_type>
Nystrom<float_type>::~Nystrom(void) {
}

template<typename float_type>
float_type Nystrom<float_type>::GetError(const Data<float_type>* data) {
    if (data->G().cols() != 0) {
        MatrixType Gtilde;
        if (have_svd_) {
            Gtilde = U_ * Sigma_.asDiagonal() * U_.transpose();
        }  else {
            Gtilde = Ctransp_.transpose() * Winv_ * Ctransp_;
        }
        return (data->G()-Gtilde).norm()/(data->G().norm());
    } else {
        return float_type(-1.0);
    }
}

/*template<typename float_type>
std::vector<uint64_t> Nystrom<float_type>::Sample(const Data<float_type> *data, MatrixType &W, RowMatrixType &Ctransp, const Data<float_type> *data)
{
    std::vector<uint64_t> Lambda;
    const uint64_t k = W.cols();
    uint64_t nitems = data->num_items();

    {
        RuntimeMonitorScope scope (*runtime_, "Choose ", k, " columns");
        std::random_device random_device;
        std::mt19937 rnd(random_device());
        for (uint64_t i = 0; i < k; i++) {
            uint64_t col = std::uniform_int_distribution<uint64_t> (0, nitems - k) (rnd) + i;
            Lambda.push_back(col);
        }

    }


    {
        RuntimeMonitorScope scope (*runtime_, "Compute C_k^T");
        uint64_t j = 0;
        for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
            Ctransp.row(j++) = data->column(*it);
        }
    }

    {
        RuntimeMonitorScope scope (*runtime_, "Fetch W");
        uint64_t i = 0;
        for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
            W.row(i) = Ctransp_.col(*it);
            i++;
        }
    }

    return Lambda;
}*/

template<typename float_type>
void Nystrom<float_type>::ScaleSVD (RowVectorType &sngVals, MatrixType &sngVecs, MatrixType const& W, RowMatrixType const& Ctransp,
  const uint64_t nitems, const std::shared_ptr<RuntimeMonitor> &runtime) {
    uint64_t k = W.rows();

#ifdef USE_BDCSVD
    Eigen::BDCSVD<MatrixType> SVD;
#else
    #warning "BDCSVD not available. Falling back to JacobiSVD"
    Eigen::JacobiSVD<MatrixType, Eigen::NoQRPreconditioner> SVD;
#endif
    {
        RuntimeMonitorScope scope (*runtime, "Compute SVD of W");
        SVD = SVD.compute (W, Eigen::ComputeThinU);
    }

    int small_singular_values = 0;
    float_type cutoff = float_type(1e2)*std::numeric_limits<float_type>::epsilon();
    {
        RuntimeMonitorScope scope (*runtime, "Scale to SVD of Gtilde");
        sngVals = (float_type(nitems) / float_type(k)) * SVD.singularValues();
        auto sngValInv = SVD.singularValues();
        for (auto i = 0; i < sngValInv.rows(); i++) {
            float_type &v = sngValInv(i);
            if (my_abs(v)<=cutoff) {
                small_singular_values++;
                v = float_type(0.0);
            } else {
                v = (float_type(1.0)/v);
            }
        }
        sngVecs = std::sqrt(float_type(k) / float_type(nitems)) * Ctransp.transpose() * SVD.matrixU() * sngValInv.asDiagonal();
    }
    if (small_singular_values && Arguments::get().verbose()) {
        std::cout << "  (" << small_singular_values << " singular values < " << cutoff << " were cut off)" << std::endl;
    }
}

template class Nystrom<float>;
template class Nystrom<double>;
template class Nystrom<long double>;
#ifdef USE_MPFR
template class Nystrom<mpfr::mpreal>;
#endif
