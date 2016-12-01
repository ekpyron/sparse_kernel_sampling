#include "PseudoInv.hpp"
#include <utility/Arguments.hpp>
#include <utility/RuntimeMonitor.hpp>
#include <utility/mymath.hpp>
#include <limits>
#include <iostream>
#ifdef USE_MPFR
#include <mpreal.h>
#endif

template<typename float_type>
typename PseudoInverse<float_type>::MatrixType PseudoInverse<float_type>::compute(MatrixType const &M, const std::shared_ptr<RuntimeMonitor> &runtime)
{
    MatrixType result;
#ifdef USE_BDCSVD
    Eigen::BDCSVD<MatrixType> SVD;
#else
    #warning "BDCSVD not available. Falling back to JacobiSVD"
    Eigen::JacobiSVD<MatrixType, Eigen::NoQRPreconditioner> SVD;
#endif
    {
        RuntimeMonitorScope scope (*runtime, "Compute SVD");
        SVD = SVD.compute(M, Eigen::ComputeThinU);
    }

    int small_singular_values = 0;
    float_type cutoff = float_type(1e2)*std::numeric_limits<float_type>::epsilon();
    {
        RuntimeMonitorScope scope (*runtime, "Compute pseudo-inverse (", SVD.nonzeroSingularValues(), ")");
        auto singValInv = SVD.singularValues();
        for (auto i = 0; i < singValInv.rows(); i++) {
            float_type &v = singValInv(i);
            if (my_abs(v)<=cutoff) {
                small_singular_values++;
                v = float_type(0.0);
            } else {
                v = (float_type(1.0)/v);
            }
        }
        result = SVD.matrixU() * (singValInv.asDiagonal()) * SVD.matrixU().transpose();
    }
    if (small_singular_values && Arguments::get().verbose()) {
        std::cout << "  (" << small_singular_values << " singular values < " << cutoff << " were cut off)" << std::endl;
    }
    return result;
}

template class PseudoInverse<float>;
template class PseudoInverse<double>;
template class PseudoInverse<long double>;
#ifdef USE_MPFR
//template class PseudoInverse<mpfr::mpreal>;
#endif
