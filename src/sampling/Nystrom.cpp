#include "Nystrom.hpp"
#include <eigen3/Eigen/Eigen>
#include <random>
#include <iostream>
#ifdef USE_MPFR
#include <mpreal.h>
#endif
#include <utility/mymath.hpp>
#include <utility/Arguments.hpp>

template<typename float_type>
Nystrom<float_type>::Nystrom(const Data<float_type>* data, const uint64_t k, const std::shared_ptr<RuntimeMonitor> &runtime)
        : Ctransp_(k, data->num_items()), Winv_(k, k), k_ (k), runtime_(runtime) {
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

    MatrixType W (k, k);
    {
        RuntimeMonitorScope scope (*runtime_, "Fetch W");
        uint64_t i = 0;
        for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
            W.row(i) = Ctransp_.col(*it);
            i++;
        }
    }

    Eigen::JacobiSVD<MatrixType> SVD;
    {
        RuntimeMonitorScope scope (*runtime_, "Compute SVD of W");
        SVD = W.jacobiSvd (Eigen::ComputeFullU|Eigen::ComputeFullV);
    }

    int small_singular_values = 0;
    {
        RuntimeMonitorScope scope (*runtime_, "Compute W^{-1} (", SVD.nonzeroSingularValues(), ")");
        auto singValInv = SVD.singularValues();
        for (auto i = 0; i < singValInv.rows(); i++) {
            float_type &v = singValInv(i);
            if (my_abs(v)<=float_type(1e-10)) {
                small_singular_values++;
                v = float_type(0.0);
            } else {
                v = (float_type(1.0)/v);
            }
        }
        Winv_ = SVD.matrixV() * (singValInv.asDiagonal()) * SVD.matrixU().transpose();
    }
    if (small_singular_values && Arguments::get().verbose()) {
        std::cout << "  (" << small_singular_values << " singular values < 1e-10 were cut off)" << std::endl;
    }
}

template<typename float_type>
Nystrom<float_type>::~Nystrom(void) {
}

template<typename float_type>
float_type Nystrom<float_type>::GetError(const Data<float_type>* data) {
    if (data->G().cols() != 0) {
        MatrixType Gtilde = Ctransp_.transpose() * Winv_ * Ctransp_;
        return (data->G()-Gtilde).norm()/(data->G().norm());
    } else {
        return float_type(-1.0);
    }
}

template class Nystrom<float>;
template class Nystrom<double>;
template class Nystrom<long double>;
#ifdef USE_MPFR
template class Nystrom<mpfr::mpreal>;
#endif
