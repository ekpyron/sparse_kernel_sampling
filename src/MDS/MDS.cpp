#include <iostream>
#include "MDS.hpp"
#ifdef USE_MPFR
#include <mpreal.h>
#endif

template<typename float_type>
MDS<float_type>::MDS(const typename MDS<float_type>::MatrixType& W) {
    size_t n = W.rows();

    const size_t k = 2;

    //MatrixType H = MatrixType::Identity(n,n) - (float_type(1) / float_type(n)) * MatrixType::Ones(n,n);
    MatrixType H = (float_type(1) / float_type(n)) * MatrixType::Ones(n,n);

    MatrixType B = (W);
    //MatrixType B = W - H*W - W*H + H*W*H;

    avg_ = MatrixType(1,n);
    //(float_type(1.0)/float_type(n)) * B.colwise().sum();

    Eigen::SelfAdjointEigenSolver<MatrixType> eigenSolver;

    auto ED = eigenSolver.compute(B, Eigen::ComputeEigenvectors);

    std::vector<std::pair<size_t, float_type>> eigenvalues (n);
    for (auto i = 0; i < n; i++) {
        eigenvalues[i] = std::make_pair(i, ED.eigenvalues()(i));
    }
    std::sort(eigenvalues.begin(), eigenvalues.end(), [](const std::pair<size_t, float_type> &lhs, const std::pair<size_t, float_type> &rhs) {
        return lhs.second > rhs.second;
    });

    MatrixType L(k,n);
    Lt_ = MatrixType (k,n);

    for (auto i = 0; i < k; i++) {
        L.row(i) = std::sqrt(eigenvalues[i].second) * ED.eigenvectors().col(eigenvalues[i].first).transpose();
        Lt_.row(i) = (float_type(1.0) / std::sqrt (eigenvalues[i].second)) * ED.eigenvectors().row(eigenvalues[i].first);
        std::cout << eigenvalues[i].second << std::endl;
    }
}

template<typename float_type>
MDS<float_type>::~MDS() {

}

template class MDS<float>;
template class MDS<double>;
template class MDS<long double>;
#ifdef USE_MPFR
template class MDS<mpfr::mpreal>;
#endif
