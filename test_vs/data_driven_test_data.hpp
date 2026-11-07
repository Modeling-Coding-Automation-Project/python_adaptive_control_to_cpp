#ifndef DATA_DRIVEN_TEST_DATA_HPP_
#define DATA_DRIVEN_TEST_DATA_HPP_

#include "data_driven.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace DataDrivenTest {

static const std::size_t samples = 256;

// Zero-state open-loop experiment. Integer PRNG makes the fixture portable
// between Python and C++ without depending on a library's RNG implementation.
template <typename T, std::size_t N>
void experiment(std::array<T, N>& u, std::array<T, N>& y, T noise = T(0)) {
    std::uint32_t state = 0x12345678u;
    T previous_y = T(0);
    T previous_u = T(0);
    for (std::size_t k = 0; k < N; ++k) {
        state = 1664525u * state + 1013904223u;
        u[k] = T(state >> 16) / T(32768) - T(1);
        const T plant_y = T(0.7) * previous_y + T(0.3) * previous_u;
        const int noise_code = static_cast<int>((k * 17 + 3) % 23) - 11;
        y[k] = plant_y + noise * T(noise_code) / T(11);
        previous_y = plant_y;
        previous_u = u[k];
    }
}

template <typename T>
PythonAdaptiveControl::RationalFilter<T, 1> model() {
    return PythonAdaptiveControl::RationalFilter<T, 1>(
        std::array<T, 2>{{T(0), T(0.25)}},
        std::array<T, 2>{{T(1), T(-0.75)}});
}

template <typename T>
PythonAdaptiveControl::LinearControllerBasis<T, 2, 1> forward_basis() {
    return PythonAdaptiveControl::LinearControllerBasis<T, 2, 1>(
        std::array<std::array<T, 2>, 2>{{
            {{T(1), T(-1)}}, {{T(0.1), T(0)}}}},
        std::array<T, 2>{{T(1), T(-1)}});
}

template <typename T>
PythonAdaptiveControl::LinearControllerBasis<T, 2, 1> inverse_basis() {
    return PythonAdaptiveControl::LinearControllerBasis<T, 2, 1>(
        std::array<std::array<T, 2>, 2>{{
            {{T(1), T(0)}}, {{T(0), T(1)}}}},
        std::array<T, 2>{{T(1), T(-0.7)}});
}

// Independently evaluate the ordinary FRIT objective using scalar recurrences,
// not the library's filtering/regression/objective helpers.
template <typename T, std::size_t N>
double independent_frit_cost(const std::array<T, N>& u,
                             const std::array<T, N>& y,
                             const std::array<T, 2>& p,
                             std::size_t discard, double ridge) {
    double previous_inverse = 0.0;
    double previous_u = 0.0;
    double previous_reference = 0.0;
    double previous_model = 0.0;
    double squared_error = 0.0;
    const double b0 = double(p[0]) + 0.1 * double(p[1]);
    for (std::size_t k = 0; k < N; ++k) {
        const double inverse = (double(u[k]) - previous_u +
            double(p[0]) * previous_inverse) / b0;
        const double reference = double(y[k]) + inverse;
        const double prediction = 0.25 * previous_reference + 0.75 * previous_model;
        const double error = double(y[k]) - prediction;
        if (k >= discard) squared_error += error * error;
        previous_inverse = inverse;
        previous_u = double(u[k]);
        previous_reference = reference;
        previous_model = prediction;
    }
    return squared_error / double(N - discard) +
        ridge * (double(p[0]) * double(p[0]) + double(p[1]) * double(p[1]));
}

// Independent closed-loop recurrence: PI uses the current error in its integral.
template <typename T, std::size_t N>
std::array<T, N> closed_loop(const std::array<T, 2>& p) {
    std::array<T, N> output = {{}};
    T integral = T(0), previous_y = T(0), previous_u = T(0);
    for (std::size_t k = 0; k < N; ++k) {
        output[k] = T(0.7) * previous_y + T(0.3) * previous_u;
        const T error = T(1) - output[k];
        integral += T(0.1) * error;
        previous_u = p[0] * error + p[1] * integral;
        previous_y = output[k];
    }
    return output;
}

} // namespace DataDrivenTest

#endif
