#ifndef PYTHON_ADAPTIVE_CONTROL_VRFT_HPP_
#define PYTHON_ADAPTIVE_CONTROL_VRFT_HPP_

#include "data_driven_filter.hpp"
#include "data_driven_least_squares.hpp"
#include "data_driven_optimization.hpp"

namespace PythonAdaptiveControl {

/** Filtered VRFT with F=M:
 * minimize mean |M u - sum rho_i beta_i (I-M)y|^2 + ridge |rho|^2.
 * This form avoids applying a noncausal M^-1 when M contains a delay.
 * An optional prefilter H is applied to BOTH target and regression columns.
 */
template <typename T, std::size_t P, std::size_t Order, std::size_t N>
Regression<T, P, N> vrft_regression(
    const std::array<T, N> &u, const std::array<T, N> &y,
    const RationalFilter<T, Order> &model,
    const LinearControllerBasis<T, P, Order> &basis,
    const RationalFilter<T, Order> &prefilter = RationalFilter<T, Order>()) {
  Regression<T, P, N> result;
  result.status = basis.validate();
  if (result.status != Status::SUCCESS) return result;
  if (!detail::all_finite(u) || !detail::all_finite(y)) {
    result.status = Status::NON_FINITE; return result;
  }
  std::array<T, N> error = {};
  result.status = model.apply(y, error);
  if (result.status != Status::SUCCESS) return result;
  for (std::size_t k = 0; k < N; ++k) error[k] = y[k] - error[k];
  result.status = model.apply(u, result.target);
  if (result.status != Status::SUCCESS) return result;
  result.status = prefilter.apply(result.target, result.target);
  if (result.status != Status::SUCCESS) return result;
  for (std::size_t j = 0; j < P; ++j) {
    const RationalFilter<T, Order> term(basis.numerators[j], basis.denominator);
    result.status = term.apply(error, result.columns[j]);
    if (result.status != Status::SUCCESS) return result;
    result.status = prefilter.apply(result.columns[j], result.columns[j]);
    if (result.status != Status::SUCCESS) return result;
  }
  return result;
}

template <typename T, std::size_t P, std::size_t Order, std::size_t N>
FitResult<T, P> vrft_least_squares(
    const std::array<T, N> &u, const std::array<T, N> &y,
    const RationalFilter<T, Order> &model,
    const LinearControllerBasis<T, P, Order> &basis,
    const SolverOptions<T> &options = SolverOptions<T>(),
    const RationalFilter<T, Order> &prefilter = RationalFilter<T, Order>()) {
  return solve_least_squares(vrft_regression(u, y, model, basis, prefilter), options);
}

template <typename T, std::size_t P, std::size_t Order, std::size_t N>
FitResult<T, P> vrft_optimize(
    const std::array<T, N> &u, const std::array<T, N> &y,
    const RationalFilter<T, Order> &model,
    const LinearControllerBasis<T, P, Order> &basis,
    const std::array<T, P> &initial, const std::array<T, P> &lower,
    const std::array<T, P> &upper,
    const SolverOptions<T> &options = SolverOptions<T>(),
    const RationalFilter<T, Order> &prefilter = RationalFilter<T, Order>()) {
  if (!detail::valid_options<T, N>(options)) return FitResult<T, P>();
  const Regression<T, P, N> regression = vrft_regression(u, y, model, basis, prefilter);
  if (regression.status != Status::SUCCESS) {
    FitResult<T, P> result;
    result.status = regression.status;
    return result;
  }
  // The small quadratic is formed once. PANOC evaluations then cost O(P^2),
  // independent of the number of experimental samples. QR remains the direct
  // solver: this Gram matrix is never inverted.
  std::array<std::array<T, P>, P> gram = {};
  std::array<T, P> linear = {};
  T constant = T(0);
  const T scale = T(1) / T(N - options.discard_samples);
  for (std::size_t k = options.discard_samples; k < N; ++k) {
    constant += regression.target[k] * regression.target[k] * scale;
    for (std::size_t i = 0; i < P; ++i) {
      linear[i] += regression.columns[i][k] * regression.target[k] * scale;
      for (std::size_t j = 0; j < P; ++j)
        gram[i][j] += regression.columns[i][k] * regression.columns[j][k] * scale;
    }
  }
  for (std::size_t j = 0; j < P; ++j) gram[j][j] += options.ridge;
  const auto objective = [&gram, &linear, constant](const std::array<T, P> &parameters) {
    CostGradient<T, P> value;
    value.cost = constant;
    for (std::size_t i = 0; i < P; ++i) {
      T product = T(0);
      for (std::size_t j = 0; j < P; ++j) product += gram[i][j] * parameters[j];
      value.gradient[i] = T(2) * (product - linear[i]);
      value.cost += parameters[i] * product - T(2) * linear[i] * parameters[i];
    }
    value.status = detail::finite(value.cost) && detail::all_finite(value.gradient)
        ? Status::SUCCESS : Status::NON_FINITE;
    return value;
  };
  FitResult<T, P> result = detail::optimize_parameters<T, P>(objective, initial, lower, upper, options);
  if (result.status == Status::SUCCESS || result.status == Status::ITERATION_LIMIT) {
    const CostGradient<T, P> final = regression_cost_gradient(regression, result.parameters, options);
    result.cost = final.cost;
    if (final.status != Status::SUCCESS) result.status = final.status;
  }
  return result;
}

template <typename T, std::size_t P, std::size_t Order> class VRFT {
public:
  RationalFilter<T, Order> reference_model;
  LinearControllerBasis<T, P, Order> basis;
  RationalFilter<T, Order> prefilter;
  VRFT(const RationalFilter<T, Order> &model,
       const LinearControllerBasis<T, P, Order> &controller_basis,
       const RationalFilter<T, Order> &weight = RationalFilter<T, Order>())
      : reference_model(model), basis(controller_basis), prefilter(weight) {}
  template <std::size_t N>
  Regression<T, P, N> regression(const std::array<T, N> &u, const std::array<T, N> &y) const {
    return vrft_regression(u, y, reference_model, basis, prefilter);
  }
  template <std::size_t N>
  FitResult<T, P> fit_least_squares(const std::array<T, N> &u, const std::array<T, N> &y,
      const SolverOptions<T> &options = SolverOptions<T>()) const {
    return vrft_least_squares(u, y, reference_model, basis, options, prefilter);
  }
  template <std::size_t N>
  FitResult<T, P> fit_iterative(const std::array<T, N> &u, const std::array<T, N> &y,
      const std::array<T, P> &initial, const std::array<T, P> &lower,
      const std::array<T, P> &upper, const SolverOptions<T> &options = SolverOptions<T>()) const {
    return vrft_optimize(u, y, reference_model, basis, initial, lower, upper, options, prefilter);
  }
};
} // namespace PythonAdaptiveControl
#endif
