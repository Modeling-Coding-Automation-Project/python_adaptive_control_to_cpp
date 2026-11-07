#ifndef PYTHON_ADAPTIVE_CONTROL_FILTER_HPP_
#define PYTHON_ADAPTIVE_CONTROL_FILTER_HPP_

#include "common.hpp"

namespace PythonAdaptiveControl {

template <typename T, std::size_t Order> class FilterState;

/** Rational transfer function in ascending powers of z^-1.
 * Order is a capacity: pad shorter numerator/denominator with trailing zeros.
 * All batch filtering starts from zero state. No causality-changing coefficient
 * cancellation is performed; a pure delay cannot be inverted causally.
 */
template <typename T, std::size_t Order> struct RationalFilter {
  static_assert(std::is_floating_point<T>::value, "Use a floating point scalar type");
  typedef std::array<T, Order + 1> Coefficients;
  Coefficients numerator;
  Coefficients denominator;

  RationalFilter() : numerator(), denominator() {
    numerator[0] = T(1);
    denominator[0] = T(1);
  }
  RationalFilter(const Coefficients &b, const Coefficients &a)
      : numerator(b), denominator(a) {}

  Status validate() const {
    if (!detail::all_finite(numerator) || !detail::all_finite(denominator))
      return Status::NON_FINITE;
    return denominator[0] == T(0) ? Status::INVALID_ARGUMENT : Status::SUCCESS;
  }

  template <std::size_t N>
  Status apply(const std::array<T, N> &input, std::array<T, N> &output) const {
    static_assert(N > 0, "A signal must contain at least one sample");
    const Status validity = validate();
    if (validity != Status::SUCCESS) return validity;
    if (!detail::all_finite(input)) return Status::NON_FINITE;
    Coefficients b, a;
    for (std::size_t j = 0; j <= Order; ++j) {
      b[j] = numerator[j] / denominator[0];
      a[j] = denominator[j] / denominator[0];
    }
    if (!detail::all_finite(b) || !detail::all_finite(a)) return Status::NON_FINITE;
    // Transposed direct form II: only Order state scalars, no heap storage.
    std::array<T, Order + 1> state = {};
    for (std::size_t k = 0; k < N; ++k) {
      const T sample = input[k]; // in-place input/output is supported
      const T value = b[0] * sample + state[0];
      if (!detail::finite(value)) return Status::NON_FINITE;
      for (std::size_t j = 0; j < Order; ++j) {
        state[j] = state[j + 1] + b[j + 1] * sample - a[j + 1] * value;
        if (!detail::finite(state[j])) return Status::NON_FINITE;
      }
      output[k] = value;
    }
    return Status::SUCCESS;
  }

  Status inverse(RationalFilter &output) const {
    const Status validity = validate();
    if (validity != Status::SUCCESS) return validity;
    if (numerator[0] == T(0)) return Status::SINGULAR_INVERSE;
    output = RationalFilter(denominator, numerator);
    return Status::SUCCESS;
  }
  FilterState<T, Order> state() const;
};

/** Persistent runtime state for executing a tuned controller one sample at a
 * time. Coefficients are normalized only at construction; step is O(Order).
 */
template <typename T, std::size_t Order> class FilterState {
public:
  explicit FilterState(const RationalFilter<T, Order> &filter)
      : b_(), a_(), state_(), status_(filter.validate()) {
    if (status_ != Status::SUCCESS) return;
    for (std::size_t j = 0; j <= Order; ++j) {
      b_[j] = filter.numerator[j] / filter.denominator[0];
      a_[j] = filter.denominator[j] / filter.denominator[0];
    }
    if (!detail::all_finite(b_) || !detail::all_finite(a_)) status_ = Status::NON_FINITE;
  }
  void reset() { state_.fill(T(0)); }
  Status status() const { return status_; }
  Status step(T input, T &output) {
    if (status_ != Status::SUCCESS) return status_;
    if (!detail::finite(input)) return Status::NON_FINITE;
    const T value = b_[0] * input + state_[0];
    if (!detail::finite(value)) return Status::NON_FINITE;
    // Commit a complete state only after every component is finite.
    std::array<T, Order + 1> next = {};
    for (std::size_t j = 0; j < Order; ++j) {
      next[j] = state_[j + 1] + b_[j + 1] * input - a_[j + 1] * value;
      if (!detail::finite(next[j])) return Status::NON_FINITE;
    }
    state_ = next;
    output = value;
    return Status::SUCCESS;
  }
private:
  std::array<T, Order + 1> b_, a_, state_;
  Status status_;
};

template <typename T, std::size_t Order>
using RationalFilterState = FilterState<T, Order>;

template <typename T, std::size_t Order>
FilterState<T, Order> RationalFilter<T, Order>::state() const {
  return FilterState<T, Order>(*this);
}

/** C(rho,z) = sum_i rho[i] * numerators[i](z) / denominator(z).
 * For FRIT least squares this represents C^-1 rather than C.
 */
template <typename T, std::size_t P, std::size_t Order>
struct LinearControllerBasis {
  static_assert(P > 0, "At least one basis element is required");
  typedef std::array<T, Order + 1> Coefficients;
  std::array<Coefficients, P> numerators;
  Coefficients denominator;
  LinearControllerBasis() : numerators(), denominator() { denominator[0] = T(1); }
  LinearControllerBasis(const std::array<Coefficients, P> &b, const Coefficients &a)
      : numerators(b), denominator(a) {}
  Status validate() const {
    if (!detail::all_finite(denominator)) return Status::NON_FINITE;
    if (denominator[0] == T(0)) return Status::INVALID_ARGUMENT;
    for (std::size_t j = 0; j < P; ++j)
      if (!detail::all_finite(numerators[j])) return Status::NON_FINITE;
    return Status::SUCCESS;
  }
  RationalFilter<T, Order> controller(const std::array<T, P> &parameters) const {
    Coefficients b = {};
    for (std::size_t p = 0; p < P; ++p)
      for (std::size_t j = 0; j <= Order; ++j)
        b[j] += parameters[p] * numerators[p][j];
    return RationalFilter<T, Order>(b, denominator);
  }
};

template <typename T, std::size_t Order>
using TransferFunction = RationalFilter<T, Order>;
template <typename T, std::size_t P, std::size_t Order>
using ControllerBasis = LinearControllerBasis<T, P, Order>;
} // namespace PythonAdaptiveControl
#endif
