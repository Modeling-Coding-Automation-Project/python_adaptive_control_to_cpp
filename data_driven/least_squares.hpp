#ifndef PYTHON_ADAPTIVE_CONTROL_LEAST_SQUARES_HPP_
#define PYTHON_ADAPTIVE_CONTROL_LEAST_SQUARES_HPP_

#include "common.hpp"
#include <algorithm>

namespace PythonAdaptiveControl {

template <typename T, std::size_t P, std::size_t N>
CostGradient<T, P> regression_cost_gradient(
    const Regression<T, P, N> &regression, const std::array<T, P> &parameters,
    const SolverOptions<T> &options = SolverOptions<T>()) {
  CostGradient<T, P> result;
  if (regression.status != Status::SUCCESS) {
    result.status = regression.status;
    return result;
  }
  if (!detail::valid_options<T, N>(options)) return result;
  if (!detail::all_finite(parameters)) { result.status = Status::NON_FINITE; return result; }
  const T scale = T(1) / T(N - options.discard_samples);
  result.cost = T(0);
  for (std::size_t k = options.discard_samples; k < N; ++k) {
    T residual = -regression.target[k];
    for (std::size_t j = 0; j < P; ++j)
      residual += regression.columns[j][k] * parameters[j];
    result.cost += residual * residual * scale;
    for (std::size_t j = 0; j < P; ++j)
      result.gradient[j] += T(2) * residual * regression.columns[j][k] * scale;
  }
  for (std::size_t j = 0; j < P; ++j) {
    result.cost += options.ridge * parameters[j] * parameters[j];
    result.gradient[j] += T(2) * options.ridge * parameters[j];
  }
  result.status = detail::finite(result.cost) && detail::all_finite(result.gradient)
      ? Status::SUCCESS : Status::NON_FINITE;
  return result;
}

namespace detail {
// Row-by-row Givens QR avoids squaring the condition number as normal equations
// do, while keeping only P(P+1) scalars of factorization workspace.
template <typename T, std::size_t P>
bool append_qr_row(std::array<std::array<T, P>, P> &r,
                   std::array<T, P> &qty, std::array<T, P> row, T target) {
  for (std::size_t j = 0; j < P; ++j) {
    const T magnitude = std::hypot(r[j][j], row[j]);
    if (!finite(magnitude)) return false;
    if (magnitude == T(0)) continue;
    const T cosine = r[j][j] / magnitude;
    const T sine = row[j] / magnitude;
    r[j][j] = magnitude;
    for (std::size_t k = j + 1; k < P; ++k) {
      const T old = r[j][k];
      r[j][k] = cosine * old + sine * row[k];
      row[k] = -sine * old + cosine * row[k];
    }
    const T old = qty[j];
    qty[j] = cosine * old + sine * target;
    target = -sine * old + cosine * target;
  }
  return all_finite(qty);
}
} // namespace detail

template <typename T, std::size_t P, std::size_t N>
FitResult<T, P> solve_least_squares(
    const Regression<T, P, N> &regression,
    const SolverOptions<T> &options = SolverOptions<T>()) {
  FitResult<T, P> result;
  if (regression.status != Status::SUCCESS) {
    result.status = regression.status;
    return result;
  }
  if (!detail::valid_options<T, N>(options)) return result;
  std::array<std::array<T, P>, P> r = {};
  std::array<T, P> qty = {};
  const T scale = T(1) / std::sqrt(T(N - options.discard_samples));
  for (std::size_t k = options.discard_samples; k < N; ++k) {
    std::array<T, P> row = {};
    for (std::size_t j = 0; j < P; ++j) row[j] = regression.columns[j][k] * scale;
    if (!detail::all_finite(row) || !detail::finite(regression.target[k]) ||
        !detail::append_qr_row(r, qty, row, regression.target[k] * scale)) {
      result.status = Status::NON_FINITE;
      return result;
    }
  }
  if (options.ridge > T(0)) {
    for (std::size_t j = 0; j < P; ++j) {
      std::array<T, P> row = {};
      row[j] = std::sqrt(options.ridge);
      if (!detail::append_qr_row(r, qty, row, T(0))) {
        result.status = Status::NON_FINITE;
        return result;
      }
    }
  }
  T largest = T(0);
  for (std::size_t j = 0; j < P; ++j) largest = std::max(largest, std::abs(r[j][j]));
  for (std::size_t j = 0; j < P; ++j)
    if (std::abs(r[j][j]) > options.rank_tolerance * largest) ++result.rank;
  if (result.rank != P) { result.status = Status::RANK_DEFICIENT; return result; }
  for (std::size_t index = P; index > 0; --index) {
    const std::size_t j = index - 1;
    T value = qty[j];
    for (std::size_t k = j + 1; k < P; ++k) value -= r[j][k] * result.parameters[k];
    result.parameters[j] = value / r[j][j];
  }
  const CostGradient<T, P> objective = regression_cost_gradient(regression, result.parameters, options);
  result.cost = objective.cost;
  result.status = objective.status;
  return result;
}
} // namespace PythonAdaptiveControl
#endif
