#ifndef PYTHON_ADAPTIVE_CONTROL_DATA_DRIVEN_LEAST_SQUARES_HPP_
#define PYTHON_ADAPTIVE_CONTROL_DATA_DRIVEN_LEAST_SQUARES_HPP_

#include "data_driven_common.hpp"
#include "python_numpy_linalg_qr.hpp"

namespace PythonAdaptiveControl {

template <typename T, std::size_t P, std::size_t N>
CostGradient<T, P>
regression_cost_gradient(const Regression<T, P, N> &regression,
                         const std::array<T, P> &parameters,
                         const SolverOptions<T> &options = SolverOptions<T>()) {
  CostGradient<T, P> result;
  if (regression.status != Status::SUCCESS) {
    result.status = regression.status;
    return result;
  }
  if (!detail::valid_options<T, N>(options))
    return result;
  if (!detail::all_finite(parameters)) {
    result.status = Status::NON_FINITE;
    return result;
  }
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
  result.status =
      detail::finite(result.cost) && detail::all_finite(result.gradient)
          ? Status::SUCCESS
          : Status::NON_FINITE;
  return result;
}

namespace detail {
// PythonControl::LeastSquares::fit always adds a free bias coefficient. That
// changes the prescribed VRFT/FRIT controller family, so use the existing
// PythonNumpy QR API directly for the no-intercept least-squares problem.
//
// Appending one observation to the compressed augmented [R | Q^T y] matrix
// needs only a (P+1)-square QR, rather than the backend's full N-square Q.
// All rotations and the final triangular solve remain backend operations.
template <typename T, std::size_t P>
bool append_qr_row(PythonNumpy::DenseMatrix_Type<T, P + 1, P + 1> &augmented,
                   const std::array<T, P> &row, T target) {
  for (std::size_t j = 0; j < P; ++j)
    augmented(P, j) = row[j];
  augmented(P, P) = target;
  // Reconstruct each time: the pinned backend retains Q between solve calls.
  PythonNumpy::LinalgSolverQR<T, P + 1, P + 1> qr;
  // Rotation denominators are scaled by the larger component internally.
  // Keep the algorithmic zero guard separate from the relative rank test.
  qr.set_division_min(std::numeric_limits<T>::min());
  qr.solve(augmented);
  augmented = qr.get_R().create_dense();
  for (std::size_t i = 0; i <= P; ++i)
    for (std::size_t j = i; j <= P; ++j)
      if (!finite(augmented(i, j)))
        return false;
  return true;
}
} // namespace detail

template <typename T, std::size_t P, std::size_t N>
FitResult<T, P>
solve_least_squares(const Regression<T, P, N> &regression,
                    const SolverOptions<T> &options = SolverOptions<T>()) {
  FitResult<T, P> result;
  if (regression.status != Status::SUCCESS) {
    result.status = regression.status;
    return result;
  }
  if (!detail::valid_options<T, N>(options))
    return result;
  PythonNumpy::DenseMatrix_Type<T, P + 1, P + 1> augmented;
  const T scale = T(1) / PythonMath::sqrt(T(N - options.discard_samples));
  for (std::size_t k = options.discard_samples; k < N; ++k) {
    std::array<T, P> row = {};
    for (std::size_t j = 0; j < P; ++j)
      row[j] = regression.columns[j][k] * scale;
    if (!detail::all_finite(row) || !detail::finite(regression.target[k]) ||
        !detail::append_qr_row<T, P>(augmented, row,
                                     regression.target[k] * scale)) {
      result.status = Status::NON_FINITE;
      return result;
    }
  }
  if (options.ridge > T(0)) {
    for (std::size_t j = 0; j < P; ++j) {
      std::array<T, P> row = {};
      // Preserve a small positive regularizer instead of applying the math
      // wrapper's default sqrt floor to its coefficient.
      row[j] = Base::Math::sqrt(options.ridge, std::numeric_limits<T>::min());
      if (!detail::append_qr_row<T, P>(augmented, row, T(0))) {
        result.status = Status::NON_FINITE;
        return result;
      }
    }
  }
  T largest = T(0);
  for (std::size_t j = 0; j < P; ++j) {
    const T diagonal = PythonMath::abs(augmented(j, j));
    if (diagonal > largest)
      largest = diagonal;
  }
  for (std::size_t j = 0; j < P; ++j)
    if (PythonMath::abs(augmented(j, j)) > options.rank_tolerance * largest)
      ++result.rank;
  if (result.rank != P) {
    result.status = Status::RANK_DEFICIENT;
    return result;
  }

  PythonNumpy::DenseMatrix_Type<T, P, P> triangular, right_hand_sides;
  for (std::size_t i = 0; i < P; ++i) {
    right_hand_sides(i, 0) = augmented(i, P);
    for (std::size_t j = i; j < P; ++j)
      triangular(i, j) = augmented(i, j);
  }
  PythonNumpy::LinalgSolverQR<T, P, P> triangular_solver;
  triangular_solver.set_division_min(std::numeric_limits<T>::min());
  triangular_solver.solve(triangular);
  // The pinned backend branches on the RHS column count when processing an
  // off-diagonal row. A P-column RHS selects the complete substitution path;
  // only column zero is populated and returned. This also handles P=1.
  const auto solution =
      triangular_solver.backward_substitution(right_hand_sides);
  for (std::size_t j = 0; j < P; ++j)
    result.parameters[j] = solution(j, 0);
  const CostGradient<T, P> objective =
      regression_cost_gradient(regression, result.parameters, options);
  result.cost = objective.cost;
  result.status = objective.status;
  return result;
}
} // namespace PythonAdaptiveControl
#endif
