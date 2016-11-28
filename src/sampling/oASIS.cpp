#include <random>
#include "oASIS.h"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <chrono>
#include "Nystrom.hpp"
#include <utility/Arguments.hpp>

oASIS::oASIS(const Data *data, const std::shared_ptr<RuntimeMonitor> &runtime)
    : runtime_(runtime)
{
    const uint64_t init_cols = 10;
    const uint64_t max_cols = 200;
    const float err_tolerance = std::numeric_limits<float>::epsilon();
    Winv_max_ = Eigen::MatrixXf (max_cols, max_cols);
    Ctransp_max_ = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> (max_cols, data->num_items ());
    k_ = init_cols;

    uint64_t nitems = data->num_items();

    {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> R_max (max_cols, nitems);
        Eigen::RowVectorXf Delta(nitems), d(nitems);

        std::vector<bool> sampled(nitems, false);

        {
            Nystrom nystrom (data, k_, runtime_);

            Ctransp_max_.topRows(k_) = nystrom.Ctransp();
            Winv_max_.topLeftCorner (k_, k_) = nystrom.Winv();
            for (auto &col : nystrom.Lambda()) {
                sampled[col] = true;
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

                float err = 0.0f;
                uint64_t i = 0;
                for (uint64_t j = 0; j < nitems; j++) {
                    if (!sampled[j]) {
                        float v = std::abs(Delta(j));
                        if (v > err) {
                            err = v; i = j;
                        }
                    }
                }
                if (err <= err_tolerance) break;

                float s = 1.0f / Delta(i);

                const Eigen::RowVectorXf &q = R.col(i);

                Ctransp_max_.row(k_) = data->column(i);

                Eigen::VectorXf sq = s * q;

                {
                    auto tmp = (q * Ctransp - Ctransp_max_.row(k_));
                    R_max.topRows(k_) += sq * tmp;
                    R_max.row(k_) = -s * tmp;
                }

                Winv_max_.topLeftCorner(k_,k_) += sq * q;
                Winv_max_.block(k_, 0, 1, k_) = -sq.transpose();
                Winv_max_.block(0, k_, k_, 1) = -sq;
                Winv_max_(k_, k_) = s;

                k_++;
                sampled[i] = true;
            }
        }
    }
}

oASIS::~oASIS(void) {
}

float oASIS::GetError(const Data* data) const {
    if (data->G().cols() != 0) {
        auto const& Ctransp = Ctransp_max_.topRows (k_);
        auto const& Winv = Winv_max_.topLeftCorner (k_, k_);
        Eigen::MatrixXf Gtilde = Ctransp.transpose () * Winv * Ctransp;
        return (data->G()-Gtilde).norm() / (data->G().norm());
    } else {
        return -1.0f;
    }
}