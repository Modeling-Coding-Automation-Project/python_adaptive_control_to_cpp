#ifndef PYTHON_ADAPTIVE_CONTROL_FRIT_HPP_
#define PYTHON_ADAPTIVE_CONTROL_FRIT_HPP_

#include "data_driven_filter.hpp"
#include "data_driven_least_squares.hpp"
#include "data_driven_optimization.hpp"

namespace PythonAdaptiveControl {

/** Exact non-iterative FRIT for the INVERSE-linear controller family.
 * C^-1(rho)=sum rho_i beta_i; target=(I-M)y, columns=M beta_i u.
 * These parameters describe C^-1 and are not forward PID gains.
 */
template <typename T, std::size_t P, std::size_t Order, std::size_t N>
Regression<T, P, N>
frit_regression(const std::array<T, N> &u, const std::array<T, N> &y,
                const RationalFilter<T, Order> &model,
                const LinearControllerBasis<T, P, Order> &inverse_basis) {
  Regression<T, P, N> result;
  result.status = inverse_basis.validate();
  if (result.status != Status::SUCCESS)
    return result;
  if (!detail::all_finite(u) || !detail::all_finite(y)) {
    result.status = Status::NON_FINITE;
    return result;
  }
  result.status = model.apply(y, result.target);
  if (result.status != Status::SUCCESS)
    return result;
  for (std::size_t k = 0; k < N; ++k)
    result.target[k] = y[k] - result.target[k];
  for (std::size_t j = 0; j < P; ++j) {
    const RationalFilter<T, Order> term(inverse_basis.numerators[j],
                                        inverse_basis.denominator);
    result.status = term.apply(u, result.columns[j]);
    if (result.status != Status::SUCCESS)
      return result;
    result.status = model.apply(result.columns[j], result.columns[j]);
    if (result.status != Status::SUCCESS)
      return result;
  }
  return result;
}

template <typename T, std::size_t P, std::size_t Order, std::size_t N>
FitResult<T, P>
frit_least_squares(const std::array<T, N> &u, const std::array<T, N> &y,
                   const RationalFilter<T, Order> &model,
                   const LinearControllerBasis<T, P, Order> &inverse_basis,
                   const SolverOptions<T> &options = SolverOptions<T>()) {
  return solve_least_squares(frit_regression(u, y, model, inverse_basis),
                             options);
}

namespace detail {
template <typename T, std::size_t P, std::size_t Order, std::size_t N>
CostGradient<T, P>
frit_objective(const std::array<T, N> &u, const std::array<T, N> &target,
               const RationalFilter<T, Order> &model,
               const LinearControllerBasis<T, P, Order> &basis,
               const std::array<T, P> &parameters,
               const SolverOptions<T> &options) {
  CostGradient<T, P> result;
  if (!all_finite(parameters)) {
    result.status = Status::NON_FINITE;
    return result;
  }
  const RationalFilter<T, Order> controller = basis.controller(parameters);
  RationalFilter<T, Order> inverse;
  result.status = controller.inverse(inverse);
  if (result.status != Status::SUCCESS)
    return result;
  std::array<T, N> error = {}, residual = {}, derivative = {};
  result.status = inverse.apply(u, error);
  if (result.status != Status::SUCCESS)
    return result;
  result.status = model.apply(error, residual);
  if (result.status != Status::SUCCESS)
    return result;
  const T scale = T(1) / T(N - options.discard_samples);
  result.cost = T(0);
  for (std::size_t k = 0; k < N; ++k) {
    residual[k] = target[k] - residual[k];
    if (k >= options.discard_samples)
      result.cost += residual[k] * residual[k] * scale;
  }
  // C=B/A, e=C^-1 u=A/B u. d e / d rho_i = -(B_i/B)e,
  // hence d[(I-M)y-M e]/d rho_i = M(B_i/B)e.
  for (std::size_t j = 0; j < P; ++j) {
    const RationalFilter<T, Order> sensitivity(basis.numerators[j],
                                               controller.numerator);
    result.status = sensitivity.apply(error, derivative);
    if (result.status != Status::SUCCESS)
      return result;
    result.status = model.apply(derivative, derivative);
    if (result.status != Status::SUCCESS)
      return result;
    for (std::size_t k = options.discard_samples; k < N; ++k)
      result.gradient[j] += T(2) * residual[k] * derivative[k] * scale;
    result.cost += options.ridge * parameters[j] * parameters[j];
    result.gradient[j] += T(2) * options.ridge * parameters[j];
  }
  result.status = finite(result.cost) && all_finite(result.gradient)
                      ? Status::SUCCESS
                      : Status::NON_FINITE;
  return result;
}
} // namespace detail

/** Original nonlinear FRIT objective for the FORWARD-linear controller family.
 * Analytic gradient is obtained by filtering sensitivities, never by finite
 * differencing. Invertibility and finite calculations are checked, but neither
 * this cost nor box bounds guarantee closed-loop stability.
 */
template <typename T, std::size_t P, std::size_t Order, std::size_t N>
CostGradient<T, P>
frit_cost_gradient(const std::array<T, N> &u, const std::array<T, N> &y,
                   const RationalFilter<T, Order> &model,
                   const LinearControllerBasis<T, P, Order> &basis,
                   const std::array<T, P> &parameters,
                   const SolverOptions<T> &options = SolverOptions<T>()) {
  CostGradient<T, P> result;
  if (!detail::valid_options<T, N>(options))
    return result;
  result.status = basis.validate();
  if (result.status != Status::SUCCESS)
    return result;
  if (!detail::all_finite(u) || !detail::all_finite(y)) {
    result.status = Status::NON_FINITE;
    return result;
  }
  std::array<T, N> target = {};
  result.status = model.apply(y, target);
  if (result.status != Status::SUCCESS)
    return result;
  for (std::size_t k = 0; k < N; ++k)
    target[k] = y[k] - target[k];
  return detail::frit_objective(u, target, model, basis, parameters, options);
}

template <typename T, std::size_t P, std::size_t Order, std::size_t N>
FitResult<T, P>
frit_optimize(const std::array<T, N> &u, const std::array<T, N> &y,
              const RationalFilter<T, Order> &model,
              const LinearControllerBasis<T, P, Order> &basis,
              const std::array<T, P> &initial, const std::array<T, P> &lower,
              const std::array<T, P> &upper,
              const SolverOptions<T> &options = SolverOptions<T>()) {
  FitResult<T, P> result;
  if (!detail::valid_options<T, N>(options))
    return result;
  result.status = basis.validate();
  if (result.status != Status::SUCCESS)
    return result;
  if (!detail::all_finite(u) || !detail::all_finite(y)) {
    result.status = Status::NON_FINITE;
    return result;
  }
  std::array<T, N> target = {};
  result.status = model.apply(y, target);
  if (result.status != Status::SUCCESS)
    return result;
  for (std::size_t k = 0; k < N; ++k)
    target[k] = y[k] - target[k];
  const auto objective = [&u, &target, &model, &basis,
                          &options](const std::array<T, P> &parameters) {
    return detail::frit_objective(u, target, model, basis, parameters, options);
  };
  return detail::optimize_parameters<T, P>(objective, initial, lower, upper,
                                           options);
}

template <typename T, std::size_t P, std::size_t Order> class FRIT {
public:
  RationalFilter<T, Order> reference_model;
  LinearControllerBasis<T, P, Order> basis;
  FRIT(const RationalFilter<T, Order> &model,
       const LinearControllerBasis<T, P, Order> &controller_basis)
      : reference_model(model), basis(controller_basis) {}
  template <std::size_t N>
  CostGradient<T, P>
  cost_gradient(const std::array<T, P> &parameters, const std::array<T, N> &u,
                const std::array<T, N> &y,
                const SolverOptions<T> &options = SolverOptions<T>()) const {
    return frit_cost_gradient(u, y, reference_model, basis, parameters,
                              options);
  }
  template <std::size_t N>
  FitResult<T, P>
  fit_iterative(const std::array<T, N> &u, const std::array<T, N> &y,
                const std::array<T, P> &initial, const std::array<T, P> &lower,
                const std::array<T, P> &upper,
                const SolverOptions<T> &options = SolverOptions<T>()) const {
    return frit_optimize(u, y, reference_model, basis, initial, lower, upper,
                         options);
  }
};

template <typename T, std::size_t P, std::size_t Order> class FRITLeastSquares {
public:
  RationalFilter<T, Order> reference_model;
  LinearControllerBasis<T, P, Order> inverse_basis;
  FRITLeastSquares(
      const RationalFilter<T, Order> &model,
      const LinearControllerBasis<T, P, Order> &controller_inverse_basis)
      : reference_model(model), inverse_basis(controller_inverse_basis) {}
  template <std::size_t N>
  Regression<T, P, N> regression(const std::array<T, N> &u,
                                 const std::array<T, N> &y) const {
    return frit_regression(u, y, reference_model, inverse_basis);
  }
  template <std::size_t N>
  FitResult<T, P> fit_least_squares(
      const std::array<T, N> &u, const std::array<T, N> &y,
      const SolverOptions<T> &options = SolverOptions<T>()) const {
    return frit_least_squares(u, y, reference_model, inverse_basis, options);
  }
};
} // namespace PythonAdaptiveControl
#endif
