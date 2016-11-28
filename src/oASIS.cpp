#include <random>
#include "oASIS.h"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <chrono>
#include "Nystrom.hpp"

oASIS::oASIS(const Data *data, const uint64_t init_cols, const uint64_t max_cols, const float err_tolerance)
    : Winv_max_(max_cols, max_cols), Ctransp_max_ (max_cols, data->num_items ()), k_ (init_cols)
{
    uint64_t nitems = data->num_items();

    {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> R_max (max_cols, nitems);
        Eigen::RowVectorXf Delta(nitems), d(nitems);

        std::cout << "Fetching diagonal..." << std::endl;
        std::vector<bool> sampled(nitems, false);
        d = data->diagonal();
        std::cout << "DONE." << std::endl;

        std::cout << "Choose initial " << k_ << " columns..." << std::endl;
        std::vector<uint64_t> Lambda;
        std::random_device random_device;
        std::mt19937 rnd(random_device());
        for (uint64_t i = 0; i < k_; i++) {
            uint64_t col = std::uniform_int_distribution<uint64_t> (0, nitems - k_) (rnd) + i;
            Lambda.push_back(col);
            sampled[col] = true;

        }
        std::cout << "DONE." << std::endl;


        {
            std::cout << "Computing C_k^T..." << std::endl;
            uint64_t j = 0;
            for (auto it = Lambda.begin (); it != Lambda.end(); it++) {
                Ctransp_max_.row(j++) = data->column(*it);
            }
            std::cout << "DONE" << std::endl;

            std::cout << "Fetching W..." << std::endl;
            Eigen::MatrixXf W (k_, k_);
            auto const& Ctransp = Ctransp_max_.topRows(k_);
            {
                uint64_t i = 0;
                for (auto it = Lambda.begin (); it != Lambda.end(); it++) {
                    W.row(i) = Ctransp.col(*it);
                    i++;
                }
            }
            std::cout << "DONE" << std::endl;

            std::cout << "Computing SVD of W..." << std::endl;
            auto SVD = Eigen::JacobiSVD<Eigen::MatrixXf> (W, Eigen::ComputeThinU|Eigen::ComputeThinV);

            std::cout << "DONE" << std::endl;

            std::cout << "Computing W^{-1}..." << std::endl;
            auto Winv = SVD.matrixV() * (SVD.singularValues()
                    .unaryExpr([](float v)->float { return (v==0.0f)?0.0f:(1.0f/v); })
                    .asDiagonal()) * SVD.matrixU().transpose();
            Winv_max_.topLeftCorner (k_, k_) = Winv;
            std::cout << "DONE" << std::endl;

            std::cout << "Computing R..." << std::endl;
            R_max.topRows (k_) = Winv * Ctransp;
            std::cout << "DONE" << std::endl;

        }

        std::cout << "Running oASIS..." << std::endl;
        {
            RuntimeMonitorScope runtimeScope (runtime_);
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
        std::cout << "DONE." << std::endl;
        std::cout << "Runtime: " << runtime_.get().count() << " s" << std::endl;
    }
}

oASIS::~oASIS(void) {
}

void oASIS::CheckResult(const Data* data) const {
    if (data->G().cols() != 0) {
        std::cout << "Check result..." << std::endl;
        auto const& Ctransp = Ctransp_max_.topRows (k_);
        auto const& Winv = Winv_max_.topLeftCorner (k_, k_);
        Eigen::MatrixXf Gtilde = Ctransp.transpose () * Winv * Ctransp;
        std::cout << "Gtilde(" << Gtilde.rows() << ", " << Gtilde.cols () << ")" << std::endl;

        std::cout << "Error: " << (data->G()-Gtilde).norm() / (data->G().norm()) << std::endl;
        std::cout << "DONE." << std::endl;
    } else {
    }
}