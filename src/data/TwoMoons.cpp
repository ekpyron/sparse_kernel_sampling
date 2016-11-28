#include "TwoMoons.hpp"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <cmath>
#include <utility/Arguments.hpp>

TwoMoons::TwoMoons(void) {
    bool verbose = Arguments::get().verbose();

    if (verbose) std::cout << "  Read two moon data..." << std::endl;
    std::ifstream f ("two_moons", std::ios_base::in);
    if (!f.is_open ()) throw std::runtime_error("Cannot open two moon dataset.");
    std::string line;
    std::vector<Eigen::Vector2f> data;
    while (std::getline(f, line).good()) {
        char c;
        float f[2];
        std::sscanf(line.c_str(), "%f %f", &f[0], &f[1]);
        data.emplace_back(f[0], f[1]);
    }
    if (verbose) std::cout << "  DONE." << std::endl;
    if (verbose) std::cout << "  Compute Gram Matrix..." << std::endl;
    G_.resize(data.size(), data.size());

    float max_dist = 0.0f;
    for (auto i = 0; i < data.size (); i++) {
        for (auto j = i+1; j < data.size(); j++) {
            float d = (data[i] - data[j]).squaredNorm();
            if (d > max_dist) max_dist = d;
        }
    }
    float two_sigma_squared = 0.25f * max_dist;
    for (auto i = 0; i < data.size(); i++) {
        for (auto j = 0; j < data.size(); j++) {
            float dist = (data[i] - data[j]).squaredNorm();
            G_(i,j) = std::exp(-dist / two_sigma_squared);
        }
    }
    G_ += 0.0001f * Eigen::MatrixXf::Identity(G_.rows(), G_.cols());
    if (verbose) std::cout << "  DONE." << std::endl;
}

TwoMoons::~TwoMoons(void) {
}