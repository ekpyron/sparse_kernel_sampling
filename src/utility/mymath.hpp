#ifndef SPARSE_KERNEL_SAMPLING_MYEXP_HPP
#define SPARSE_KERNEL_SAMPLING_MYEXP_HPP

#include <cmath>
#ifdef USE_MPFR
#include <mpreal.h>
#endif

namespace detail {

template<typename float_type>
struct my_exp {
    static float_type exp(float_type v) {
        return std::exp(v);
    }
};

template<typename float_type>
struct my_abs {
    static float_type abs(float_type v) {
        return std::abs(v);
    }
};

#ifdef USE_MPFR
template<>
struct my_exp<mpfr::mpreal> {
    static mpfr::mpreal exp(mpfr::mpreal const& v) {
        return mpfr::exp(v);
    }
};
template<>
struct my_abs<mpfr::mpreal> {
    static mpfr::mpreal abs(mpfr::mpreal const& v) {
        return mpfr::abs(v);
    }
};
#endif

} /* namespace detail */

template<typename float_type>
inline float_type my_abs(const float_type &t) {
    return detail::my_abs<float_type>::abs(t);
}

template<typename float_type>
inline float_type my_exp(const float_type &t) {
    return detail::my_exp<float_type>::exp(t);
}


#endif /* !defined SPARSE_KERNEL_SAMPLING_MYEXP_HPP */
