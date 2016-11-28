#include "TwoMoons.hpp"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <cmath>
#include <utility/Arguments.hpp>

namespace {
template<typename float_type>
struct parse_data;

template<>
struct parse_data<float> {
    static void parse(const std::string &line, Eigen::Matrix<float, 1, 2> &data) {
        std::sscanf(line.c_str(), "%f %f", &data(0), &data(1));
    }
};

template<>
struct parse_data<double> {
    static void parse(const std::string &line, Eigen::Matrix<double, 1, 2> &data) {
        std::sscanf(line.c_str(), "%lf %lf", &data(0), &data(1));
    }
};
}

template<typename float_type>
TwoMoons<float_type>::TwoMoons(void) {
    bool verbose = Arguments::get().verbose();

    if (verbose) std::cout << "  Read two moon data..." << std::endl;
    std::ifstream f ("two_moons", std::ios_base::in);
    if (!f.is_open ()) throw std::runtime_error("Cannot open two moon dataset.");
    std::string line;
    std::vector<Eigen::Matrix<float_type, 1, 2>> data;
    while (std::getline(f, line).good()) {
        char c;
        data.emplace_back();
        parse_data<float_type>::parse(line, data.back());
    }
    if (verbose) std::cout << "  DONE." << std::endl;
    if (verbose) std::cout << "  Compute Gram Matrix..." << std::endl;
    G_.resize(data.size(), data.size());

    float_type max_dist = 0.0;
    for (auto i = 0; i < data.size (); i++) {
        for (auto j = i+1; j < data.size(); j++) {
            float_type d = (data[i] - data[j]).squaredNorm();
            if (d > max_dist) max_dist = d;
        }
    }
    float_type two_sigma_squared = float_type(0.25) * max_dist;
    for (auto i = 0; i < data.size(); i++) {
        for (auto j = 0; j < data.size(); j++) {
            float dist = (data[i] - data[j]).squaredNorm();
            G_(i,j) = std::exp(-dist / two_sigma_squared);
        }
    }
    if (verbose) std::cout << "  DONE." << std::endl;
}

template<typename float_type>
TwoMoons<float_type>::~TwoMoons(void) {
}

template class TwoMoons<float>;
template class TwoMoons<double>;
