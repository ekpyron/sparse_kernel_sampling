#include "Nystrom.hpp"
#include <eigen3/Eigen/Eigen>
#include <random>
#include <iostream>

Nystrom::Nystrom(const Data* data, const uint64_t k_) : Ctransp_(k_, data->num_items()), Winv_(k_,k_) {
    uint64_t nitems = data->num_items();

    uint64_t k = k_;

    std::cout << "NYSTROM: Choose " << k << " columns..." << std::endl;
    std::random_device random_device;
    std::mt19937 rnd(random_device());
    for (uint64_t i = 0; i < k; i++) {
        uint64_t col = std::uniform_int_distribution<uint64_t> (0, nitems - k) (rnd) + i;
        Lambda_.push_back(col);
    }
    std::cout << "NYSTROM: DONE." << std::endl;


    std::cout << "NYSTROM: Computing C_k^T..." << std::endl;
    uint64_t j = 0;
    for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
        Ctransp_.row(j++) = data->column(*it);
    }
    std::cout << "NYSTROM: DONE" << std::endl;

    std::cout << "NYSTROM: Fetching W..." << std::endl;
    Eigen::MatrixXf W (k, k);
    {
        uint64_t i = 0;
        for (auto it = Lambda_.begin (); it != Lambda_.end(); it++) {
            W.row(i) = Ctransp_.col(*it);
            i++;
        }
    }
    std::cout << "NYSTROM: DONE" << std::endl;

    std::cout << "NYSTROM: Computing SVD of W..." << std::endl;
    auto SVD = Eigen::JacobiSVD<Eigen::MatrixXf> (W, Eigen::ComputeThinU|Eigen::ComputeThinV);
    std::cout << "NYSTROM: DONE" << std::endl;

    std::cout << "NYSTROM: Computing W^{-1}..." << std::endl;
    Winv_ = SVD.matrixV() * (SVD.singularValues()
            .unaryExpr([](float v)->float { return (v==0.0f)?0.0f:(1.0f/v); })
            .asDiagonal()) * SVD.matrixU().transpose();
    std::cout << "NYSTROM: DONE" << std::endl;
}

void Nystrom::CheckResult(const Data* data) {
    if (data->G().cols() != 0) {
        std::cout << "Check result..." << std::endl;
        Eigen::MatrixXf Gtilde = Ctransp_.transpose() * Winv_ * Ctransp_;
        std::cout << "Gtilde(" << Gtilde.rows() << ", " << Gtilde.cols() << ")" << std::endl;

        std::cout << "Error: " << (data->G()-Gtilde).norm()/(data->G().norm()) << std::endl;
        std::cout << "DONE." << std::endl;
    } else {
    }
}

Nystrom::~Nystrom(void) {

}
