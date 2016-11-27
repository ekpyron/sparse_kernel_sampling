#include "Abalone.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Eigen>
#include <cmath>

Abalone::Abalone (int argc, char **argv) {
    std::cout << "Read abalone data..." << std::endl;
    std::ifstream f ("abalone.data", std::ios_base::in);
    if (!f.is_open ()) throw std::runtime_error("Cannot open abalone dataset.");
    std::string line;
    std::vector<Eigen::Matrix<float, 8, 1>> data;
    while (std::getline(f, line).good()) {
        char c;
        data.emplace_back();
        std::sscanf(line.c_str(), "%c %f %f %f %f %f %f %f %f", &c, &data.back()(0), &data.back()(1), &data.back()(2),
                    &data.back()(3), &data.back()(4), &data.back()(5), &data.back()(6), &data.back()(7));
    }
    std::cout << "DONE." << std::endl;
    std::cout << "Compute Gram Matrix..." << std::endl;
    G_.resize (data.size(), data.size());

    float max_dist = 0.0f;
    for (auto i = 0; i < data.size (); i++) {
        for (auto j = i+1; j < data.size(); j++) {
            float d = (data[i] - data[j]).norm();
            if (d > max_dist) max_dist = d;
        }
    }

    float two_sigma_squared = 2.0f * (0.25f * max_dist * max_dist);
    for (auto i = 0; i < data.size(); i++) {
        for (auto j = 0; j < data.size(); j++) {
            float dist = (data[i] - data[j]).norm();
            G_(i,j) = std::exp(-dist / two_sigma_squared);
        }
    }
    std::cout << "DONE." << std::endl;
}

Abalone::~Abalone (void) {

}