#include "Abalone.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Eigen>
#include <cmath>
#include <utility/Arguments.hpp>
#ifdef USE_MPFR
#include <sstream>
#include <mpreal.h>
#endif
#include <utility/mymath.hpp>

namespace {
template<typename float_type>
struct parse_data;

template<>
struct parse_data<float> {
    static void parse(const std::string &line, Eigen::Matrix<float, 8, 1> &data) {
        char c;
        std::sscanf(line.c_str(), "%c %f %f %f %f %f %f %f %f", &c, &data(0), &data(1), &data(2),
                    &data(3), &data(4), &data(5), &data(6), &data(7));
    }
};

template<>
struct parse_data<double> {
    static void parse(const std::string &line, Eigen::Matrix<double, 8, 1> &data) {
        char c;
        std::sscanf(line.c_str(), "%c %lf %lf %lf %lf %lf %lf %lf %lf", &c, &data(0), &data(1), &data(2),
                    &data(3), &data(4), &data(5), &data(6), &data(7));
    }
};

template<>
struct parse_data<long double> {
    static void parse(const std::string &line, Eigen::Matrix<long double, 8, 1> &data) {
        char c;
        std::sscanf(line.c_str(), "%c %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf", &c, &data(0), &data(1), &data(2),
                    &data(3), &data(4), &data(5), &data(6), &data(7));
    }
};

#ifdef USE_MPFR
template<>
struct parse_data<mpfr::mpreal> {
    static void parse(const std::string &line, Eigen::Matrix<mpfr::mpreal, 8, 1> &mpreal_data) {
        std::istringstream stream(line);
        std::string number;
        std::getline(stream, number, ' ');
        int i = 0;
        while (std::getline(stream, number, ' ') && i < 8) {
            mpreal_data(i++) = mpfr::mpreal(number);
        }
    }
};
#endif

}

template<typename float_type>
Abalone<float_type>::Abalone (void) {
    bool verbose = Arguments::get().verbose();
    if (verbose) std::cout << "  Read abalone data..." << std::endl;
    std::ifstream f ("abalone.data", std::ios_base::in);
    if (!f.is_open ()) throw std::runtime_error("Cannot open abalone dataset.");
    std::string line;
    std::vector<Eigen::Matrix<float_type, 8, 1>> data;
    while (std::getline(f, line).good()) {
        char c;
        data.emplace_back();
        parse_data<float_type>::parse(line, data.back());
    }
    if (verbose) std::cout << "  DONE." << std::endl;
    if (verbose) std::cout << "  Compute Gram Matrix..." << std::endl;
    G_.resize (data.size(), data.size());

    float_type max_dist = 0.0f;
    for (auto i = 0; i < data.size (); i++) {
        for (auto j = i+1; j < data.size(); j++) {
            float_type d = (data[i] - data[j]).squaredNorm();
            if (d > max_dist) max_dist = d;
        }
    }

    float_type two_sigma_squared = 0.25f * max_dist;
    for (auto i = 0; i < data.size(); i++) {
        for (auto j = 0; j < data.size(); j++) {
            float_type dist = (data[i] - data[j]).squaredNorm();
            G_(i,j) = my_exp(-dist / two_sigma_squared);
        }
    }
    if (verbose) std::cout << "  DONE." << std::endl;
}

template<typename float_type>
Abalone<float_type>::~Abalone (void) {
}

template class Abalone<float>;
template class Abalone<double>;
template class Abalone<long double>;
#ifdef USE_MPFR
template class Abalone<mpfr::mpreal>;
#endif
