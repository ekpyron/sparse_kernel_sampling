#include <random>
#include "oASIS.h"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <chrono>

oASIS::oASIS(Data *data, const uint64_t init_cols, const uint64_t max_cols, const float err_tolerance) {
    uint64_t nitems = data->num_items();
    std::random_device random_device;
    std::mt19937 rnd(random_device());

    uint64_t k = init_cols;

    std::cout << "Fetching diagonal..." << std::endl;
    Eigen::RowVectorXf d (nitems);
    std::vector<bool> sampled(nitems);
    for (uint64_t i = 0; i < nitems; i++) {
        d(i) = data->distance(i,i);
        sampled[i] = false;
    }
    std::cout << "DONE" << std::endl;

    std::cout << "Choose initial " << k << " columns..." << std::endl;
    std::vector<uint64_t> Lambda;
    for (uint64_t i = 0; i < k; i++) {
        uint64_t col = std::uniform_int_distribution<uint64_t> (0, nitems - k) (rnd) + i;
        Lambda.push_back(col);
        sampled[col] = true;

    }
    std::cout << "DONE." << std::endl;


    std::cout << "Computing C_k^T..." << std::endl;
    Eigen::MatrixXf Ctransp (k, nitems);
    for (auto i = 0; i < nitems; i++) {
        uint64_t j = 0;
        for (auto it = Lambda.begin (); it != Lambda.end(); it++) {
            Ctransp(j++, i) = data->distance(i, *it);
        }
    }
    std::cout << "DONE" << std::endl;

    std::cout << "Fetching W..." << std::endl;
    Eigen::MatrixXf W (k, k);
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
    Eigen::MatrixXf Winv = SVD.matrixV() * (SVD.singularValues()
            .unaryExpr([](float v)->float { return (v==0.0f)?0.0f:(1.0f/v); })
            .asDiagonal()) * SVD.matrixU().transpose();
    std::cout << "DONE" << std::endl;

    std::cout << "Computing R..." << std::endl;
    Eigen::MatrixXf R = Winv * Ctransp;
    std::cout << "DONE" << std::endl;

    std::cout << "Running oASIS...";
    std::cout.flush();
    Eigen::RowVectorXf Delta(nitems);
    Eigen::RowVectorXf c(nitems);

    auto start_time = std::chrono::steady_clock::now();
    while (k < max_cols) {
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

        for (uint64_t j = 0; j < nitems; j++) {
            c(j) = data->distance(j, i);
        }

        Eigen::VectorXf sq = s * q;

        R += sq * (q * Ctransp - c);
        R.conservativeResize(k+1, Eigen::NoChange);
        R.row(k) = s * (-q * Ctransp + c);

        Winv += (sq * q);
        Winv.conservativeResize(k+1, k+1);
        Winv.block(k, 0, 1, k) = -sq.transpose();
        Winv.block(0, k, k, 1) = -sq;
        Winv(k,k) = s;

        Ctransp.conservativeResize(k + 1, Eigen::NoChange);
        Ctransp.row(k) = c;

        k++;
        if (sampled[i]) throw std::runtime_error("Sampled twice!");
        sampled[i] = true;
    }
    auto end_time = std::chrono::steady_clock::now();
    std::cout << "DONE." << std::endl;
    std::cout << "Runtime: " << std::chrono::duration <double>(end_time-start_time).count() << " s" << std::endl;

    if (data->G().cols() != 0) {
        std::cout << "Check result..." << std::endl;
        Eigen::MatrixXf Gtilde = Ctransp.transpose() * Winv * Ctransp;
        std::cout << "Gtilde(" << Gtilde.rows() << ", " << Gtilde.cols() << ")" << std::endl;

        std::cout << "Error: " << (data->G()-Gtilde).norm()/(data->G().norm()) << std::endl;
        std::cout << "DONE." << std::endl;
    } else {
    }
}

oASIS::~oASIS(void) {
}
