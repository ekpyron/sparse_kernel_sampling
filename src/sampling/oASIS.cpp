#include <random>
#include "oASIS.h"
#include <Eigen/Eigen>
#include <iostream>
#include <chrono>
#include "Nystrom.hpp"
#include <utility/Arguments.hpp>
#include <utility/mymath.hpp>
#ifdef USE_MPFR
#include <mpreal.h>
#endif

template<typename float_type>
oASIS<float_type>::oASIS(const Data<float_type> *data, const std::shared_ptr<RuntimeMonitor> &runtime)
    : runtime_(runtime)
{
    const uint64_t init_cols = 10;
    const uint64_t max_cols = 1000;
    const float_type err_tolerance = 1e-3;//std::numeric_limits<float>::epsilon();
    {
        RuntimeMonitorScope scope(*runtime_, "Memory allocation");
        Winv_max_ = MatrixType (max_cols, max_cols);
        Ctransp_max_ = RowMatrixType (max_cols, data->num_items ());
    }
    k_ = init_cols;

    uint64_t nitems = data->num_items();

    {
        MatrixType R_max;
        RowVectorType Delta, d;
        std::vector<bool> sampled;
        {
            RuntimeMonitorScope scope(*runtime_, "Memory allocation");
            sampled.resize (nitems, false);
            R_max = MatrixType (max_cols, nitems);
            Delta = RowVectorType (nitems);
            d = RowVectorType (nitems);
        }

        {
            Nystrom<float_type> nystrom (data, k_, false, runtime_);

            {
                RuntimeMonitorScope scope(*runtime_, "Copy");
                Ctransp_max_.topRows(k_) = nystrom.Ctransp();
                Winv_max_.topLeftCorner (k_, k_) = nystrom.Winv();
                Lambda_ = nystrom.Lambda();
                for (auto &col : Lambda_) {
                    sampled[col] = true;
                }
            }

            {
                RuntimeMonitorScope scope(*runtime_, "Compute R");
                R_max.topRows (k_) = nystrom.Winv() * nystrom.Ctransp();
            }
        }

        {
            RuntimeMonitorScope scope(*runtime_, "Fetch diagonal");
            d = data->diagonal();
        }

        {
            RuntimeMonitorScope runtimeScope (*runtime_, "Run oASIS");
            while (k_ < max_cols) {
                auto const& Ctransp = Ctransp_max_.topRows (k_);
                auto const& Winv = Winv_max_.topLeftCorner (k_, k_);
                auto const& R = R_max.topRows (k_);
                Delta = d - Ctransp.cwiseProduct(R).colwise().sum();

                float_type err = float_type(0.0);
                uint64_t i = 0;
                for (uint64_t j = 0; j < nitems; j++) {
                    if (!sampled[j]) {
                        float_type v = my_abs(Delta(j));
                        if (v > err) {
                            err = v; i = j;
                        }
                    }
                }
                if (err <= err_tolerance) break;

                if (Arguments::get().verbose() && !(k_&0x1F)) {
                    RuntimeMonitorScopeSuspend suspend (runtimeScope);
                    std::cout << "    [k: " << k_ << " err: " << err << "]" << std::endl;
                }

                float s = float_type(1.0) / Delta(i);

                const RowVectorType &q = R.col(i);

                Ctransp_max_.row(k_) = data->column(i);

                VectorType sq = s * q;

                Winv_max_.topLeftCorner(k_,k_) += sq * q;
                Winv_max_.block(k_, 0, 1, k_) = -sq.transpose();
                Winv_max_.block(0, k_, k_, 1) = -sq;
                Winv_max_(k_, k_) = s;

                {
                    auto tmp = (q * Ctransp - Ctransp_max_.row(k_));
                    R_max.topRows(k_) += sq * tmp;
                    R_max.row(k_) = -s * tmp;
                }

                k_++;
                sampled[i] = true;
                Lambda_.push_back(i);
            }
        }
    }

    {
        auto const& Ctransp = Ctransp_max_.topRows (k_);
        {
            RuntimeMonitorScope scope (*runtime_, "Fetch W");
            W_ = MatrixType (k_, k_);
            uint64_t i = 0;
            for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
                W_.row(i) = Ctransp.col(*it);
                i++;
            }
        }

        Nystrom<float_type>::ScaleSVD(Sigma_, U_, W_, Ctransp, nitems, runtime_);

        std::cout << "Sigma_(" << Sigma_.rows() << ", " << Sigma_.cols() << ")" << std::endl;
        std::cout << "U_(" << U_.rows() << ", " << U_.cols() << ")" << std::endl;
        std::cout << Sigma_ << std::endl;
    }
}

template<typename float_type>
oASIS<float_type>::~oASIS(void) {
}

template<typename float_type>
float_type oASIS<float_type>::GetError(const Data<float_type>* data) const {
    if (data->G().cols() != 0) {
        MatrixType Gtilde = U_ * Sigma_.asDiagonal() * U_.transpose();
        return (data->G()-Gtilde).norm() / (data->G().norm());
    } else {
        return float_type(-1.0);
    }
}

template class oASIS<float>;
template class oASIS<double>;
template class oASIS<long double>;
#ifdef USE_MPFR
template class oASIS<mpfr::mpreal>;
#endif
