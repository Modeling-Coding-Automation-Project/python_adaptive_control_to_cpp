#ifndef PYTHON_ADAPTIVE_CONTROL_COMMON_HPP_
#define PYTHON_ADAPTIVE_CONTROL_COMMON_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace PythonAdaptiveControl {

enum class Status {
  SUCCESS,
  ITERATION_LIMIT,
  RANK_DEFICIENT,
  INVALID_ARGUMENT,
  NON_FINITE,
  SINGULAR_INVERSE
};

inline const char *status_name(Status status) {
  switch (status) {
  case Status::SUCCESS:
    return "success";
  case Status::ITERATION_LIMIT:
    return "iteration_limit";
  case Status::RANK_DEFICIENT:
    return "rank_deficient";
  case Status::INVALID_ARGUMENT:
    return "invalid_argument";
  case Status::NON_FINITE:
    return "non_finite";
  case Status::SINGULAR_INVERSE:
    return "singular_inverse";
  }
  return "invalid_argument";
}

template <typename T> struct SolverOptions {
  T ridge;
  T tolerance;
  T rank_tolerance;
  std::size_t max_iterations;
  std::size_t discard_samples;
  SolverOptions()
      : ridge(T(0)), tolerance(T(1e-8)),
        rank_tolerance(T(1e-10) > T(10) * std::numeric_limits<T>::epsilon()
                           ? T(1e-10)
                           : T(10) * std::numeric_limits<T>::epsilon()),
        max_iterations(500), discard_samples(0) {}
};

template <typename T, std::size_t P> struct FitResult {
  static_assert(P > 0, "At least one controller parameter is required");
  std::array<T, P> parameters;
  Status status;
  T cost;
  std::size_t iterations;
  std::size_t rank;
  FitResult()
      : parameters(), status(Status::INVALID_ARGUMENT),
        cost(std::numeric_limits<T>::quiet_NaN()), iterations(0), rank(0) {}
  bool has_converged() const { return status == Status::SUCCESS; }
};

template <typename T, std::size_t P> struct CostGradient {
  T cost;
  std::array<T, P> gradient;
  Status status;
  CostGradient()
      : cost(std::numeric_limits<T>::quiet_NaN()), gradient(),
        status(Status::INVALID_ARGUMENT) {}
};

template <typename T, std::size_t P, std::size_t N> struct Regression {
  static_assert(P > 0 && N > 0, "Dimensions must be positive");
  // Columns are stored separately to allow one filter pass per basis element.
  std::array<std::array<T, N>, P> columns;
  std::array<T, N> target;
  Status status;
  Regression() : columns(), target(), status(Status::INVALID_ARGUMENT) {}
};

namespace detail {
template <typename T> inline bool finite(T value) {
  return std::isfinite(value);
}
template <typename T, std::size_t N>
inline bool all_finite(const std::array<T, N> &values) {
  for (std::size_t i = 0; i < N; ++i)
    if (!finite(values[i]))
      return false;
  return true;
}
template <typename T, std::size_t N>
inline bool valid_options(const SolverOptions<T> &options) {
  static_assert(std::is_floating_point<T>::value,
                "Use float or double scalar storage");
  return finite(options.ridge) && options.ridge >= T(0) &&
         finite(options.tolerance) && options.tolerance > T(0) &&
         finite(options.rank_tolerance) && options.rank_tolerance > T(0) &&
         options.rank_tolerance < T(1) && options.max_iterations > 0 &&
         options.discard_samples < N;
}
} // namespace detail
} // namespace PythonAdaptiveControl
#endif
