#include "Nystrom.hpp"
#include <eigen3/Eigen/Eigen>
#include <random>
#include <iostream>

Nystrom::Nystrom(Data* data, const uint64_t k) : C_(data->num_items(), k), Winv_(k,k) {
    std::random_device random_device;
    std::mt19937 rnd(random_device());

    std::cout << "NYSTROM: Choose " << k << " columns..." << std::endl;
    std::vector<uint64_t> Lambda;
    for (uint64_t i = 0; i < k; i++) {
        uint64_t col = std::uniform_int_distribution<uint64_t> (0, C_.rows() - k) (rnd) + i;
        Lambda.push_back(col);

    }
    std::cout << "NYSTROM: DONE." << std::endl;


    std::cout << "NYSTROM: Computing C_k..." << std::endl;
    for (auto i = 0; i < C_.rows(); i++) {
        uint64_t j = 0;
        for (auto it = Lambda.begin (); it != Lambda.end(); it++) {
            C_(i, j++) = data->distance(i, *it);
        }
    }
    std::cout << "NYSTROM: DONE" << std::endl;

    std::cout << "NYSTROM: Fetching W..." << std::endl;
    Eigen::MatrixXf W (k, k);
    {
        uint64_t i = 0;
        for (auto it = Lambda.begin (); it != Lambda.end(); it++) {
            W(i) = C_(*it);
            i++;
        }
    }
    std::cout << "NYSTROM: DONE" << std::endl;

    std::cout << "NYSTROM: Computing SVD of W..." << std::endl;
    auto SVD = Eigen::JacobiSVD<Eigen::MatrixXf> (W, Eigen::ComputeFullU|Eigen::ComputeFullV);
    std::cout << "NYSTROM: DONE" << std::endl;

    std::cout << "NYSTROM: Computing W^{-1}..." << std::endl;
    Eigen::VectorXf singValInv (k);
    singValInv.setZero();
    for (auto i = 0; i < k; i++) {
        float sv = SVD.singularValues()(i);
        if (sv != 0.0f) singValInv(i) = 1.0f / sv;
    }
    Winv_ = SVD.matrixV() * singValInv.asDiagonal() * SVD.matrixU().transpose();
    std::cout << "NYSTROM: DONE" << std::endl;
}

Nystrom::~Nystrom(void) {

}
