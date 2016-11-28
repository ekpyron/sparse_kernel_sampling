#include "Nystrom.hpp"
#include <eigen3/Eigen/Eigen>
#include <random>
#include <iostream>

Nystrom::Nystrom(const Data* data, const uint64_t k_) : Ctransp_(k_, data->num_items()), Winv_(k_,k_) {
    uint64_t nitems = data->num_items();

    uint64_t k = k_;

    {
        RuntimeMonitorScope scope (runtime_, "Choose ", k, " columns");
        std::random_device random_device;
        std::mt19937 rnd(random_device());
        for (uint64_t i = 0; i < k; i++) {
            uint64_t col = std::uniform_int_distribution<uint64_t> (0, nitems - k) (rnd) + i;
            Lambda_.push_back(col);
        }

    }


    {
        RuntimeMonitorScope scope (runtime_, "Compute C_k^T");
        uint64_t j = 0;
        for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
            Ctransp_.row(j++) = data->column(*it);
        }
    }

    Eigen::MatrixXf W (k, k);
    {
        RuntimeMonitorScope scope (runtime_, "Fetch W");
        uint64_t i = 0;
        for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
            W.row(i) = Ctransp_.col(*it);
            i++;
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> SVD;
    {
        RuntimeMonitorScope scope (runtime_, "Compute SVD of W");
        SVD = W.jacobiSvd (Eigen::ComputeThinU|Eigen::ComputeThinV);
    }

    {
        RuntimeMonitorScope scope (runtime_, "Compute W^{-1}");
        Winv_ = SVD.matrixV() * (SVD.singularValues()
                .unaryExpr([](float v)->float { return (v==0.0f)?0.0f:(1.0f/v); })
                .asDiagonal()) * SVD.matrixU().transpose();
    }
}

Nystrom::~Nystrom(void) {
}

float Nystrom::GetError(const Data* data) {
    if (data->G().cols() != 0) {
        Eigen::MatrixXf Gtilde = Ctransp_.transpose() * Winv_ * Ctransp_;
        return (data->G()-Gtilde).norm()/(data->G().norm());
    } else {
        return -1.0f;
    }
}
