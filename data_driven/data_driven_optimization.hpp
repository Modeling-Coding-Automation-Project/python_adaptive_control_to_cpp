#ifndef PYTHON_ADAPTIVE_CONTROL_OPTIMIZATION_HPP_
#define PYTHON_ADAPTIVE_CONTROL_OPTIMIZATION_HPP_

#include "data_driven_common.hpp"
#include "python_numpy.hpp"

namespace PythonAdaptiveControl {
namespace detail {
// The upstream scalar ring buffer requires a scalar-to-vector conversion.
// All numerical operations use the existing math backend.
template <typename T, std::size_t P>
class ParameterVector : public PythonNumpy::DenseMatrix_Type<T, P, 1> {
public:
  typedef PythonNumpy::DenseMatrix_Type<T, P, 1> Base;
  ParameterVector() : Base() {}
  ParameterVector(const Base &other) : Base(other) {}
  // The upstream one-element ring buffer returns a scalar instead of a vector.
  ParameterVector(T value) : Base() { (*this)(0, 0) = value; }
  ParameterVector &operator=(const Base &other) {
    Base::operator=(other);
    return *this;
  }
};
} // namespace detail
} // namespace PythonAdaptiveControl

// Normalize the parameter vector before calling the existing square root.
// The argument then lies in [1,P], keeping the backend's small-argument floor
// from distorting PANOC's Lipschitz estimate. This uses PythonMath only, and
// does not change normalization of any existing dependency matrix type.
namespace PythonNumpy {
namespace NormalizationOperation {
template <typename T, std::size_t P>
struct Normalizer<PythonAdaptiveControl::detail::ParameterVector<T, P>,
                  std::false_type> {
  static T
  norm(const PythonAdaptiveControl::detail::ParameterVector<T, P> &matrix) {
    T scale = T(0);
    for (std::size_t j = 0; j < P; ++j) {
      if (!PythonAdaptiveControl::detail::finite(matrix(j, 0)))
        return std::numeric_limits<T>::quiet_NaN();
      const T magnitude = PythonMath::abs(matrix(j, 0));
      if (magnitude > scale)
        scale = magnitude;
    }
    if (scale == T(0))
      return T(0);
    T sum = T(0);
    for (std::size_t j = 0; j < P; ++j) {
      const T normalized = matrix(j, 0) / scale;
      sum += normalized * normalized;
    }
    return scale * PythonMath::sqrt(sum);
  }
};
} // namespace NormalizationOperation
} // namespace PythonNumpy

#include "python_optimization_panoc.hpp"

namespace PythonAdaptiveControl {
namespace detail {

// PANOC expects MPC-shaped type information. NP=1 turns the decision horizon
// into the controller parameter vector without introducing any plant model.
template <typename T, std::size_t P> struct ParameterProblem {
  static constexpr std::size_t STATE_SIZE = 1;
  static constexpr std::size_t INPUT_SIZE = P;
  static constexpr std::size_t OUTPUT_SIZE = 1;
  static constexpr std::size_t NP = 1;
  typedef T Value_Type;
  typedef PythonNumpy::DenseMatrix_Type<T, 1, 1> X_Type;
  typedef ParameterVector<T, P> U_Type;
  typedef X_Type Y_Type;
  typedef X_Type X_Horizon_Type;
  typedef U_Type U_Horizon_Type;
  typedef Y_Type Y_Horizon_Type;
  typedef PythonNumpy::DenseMatrix_Type<T, P, 1> U_Min_Type;
  typedef U_Min_Type U_Max_Type;
};

template <typename T, std::size_t P, typename Objective>
class ObjectiveAdapter {
public:
  typedef typename ParameterProblem<T, P>::U_Type Vector;
  explicit ObjectiveAdapter(const Objective &objective)
      : objective_(objective), cached_parameters_(), cached_result_(),
        cached_(false) {}
  T cost(const Vector &input) const {
    const CostGradient<T, P> value = evaluate(input);
    return value.status == Status::SUCCESS
               ? value.cost
               : std::numeric_limits<T>::quiet_NaN();
  }
  Vector gradient(const Vector &input) const {
    const CostGradient<T, P> value = evaluate(input);
    Vector output;
    for (std::size_t j = 0; j < P; ++j)
      output(j, 0) = value.status == Status::SUCCESS
                         ? value.gradient[j]
                         : std::numeric_limits<T>::quiet_NaN();
    return output;
  }

private:
  CostGradient<T, P> evaluate(const Vector &input) const {
    std::array<T, P> parameters;
    for (std::size_t j = 0; j < P; ++j)
      parameters[j] = input(j, 0);
    if (!cached_ || parameters != cached_parameters_) {
      cached_parameters_ = parameters;
      cached_result_ = objective_(parameters);
      cached_ = true;
    }
    return cached_result_;
  }
  const Objective &objective_;
  mutable std::array<T, P> cached_parameters_;
  mutable CostGradient<T, P> cached_result_;
  mutable bool cached_;
};

template <typename T, std::size_t P, typename Objective>
FitResult<T, P> optimize_parameters(const Objective &objective,
                                    const std::array<T, P> &initial,
                                    const std::array<T, P> &lower,
                                    const std::array<T, P> &upper,
                                    const SolverOptions<T> &options) {
  FitResult<T, P> result;
  if (!all_finite(initial) || !all_finite(lower) || !all_finite(upper)) {
    result.status = Status::NON_FINITE;
    return result;
  }
  for (std::size_t j = 0; j < P; ++j) {
    if (lower[j] > upper[j] || initial[j] < lower[j] || initial[j] > upper[j])
      return result;
  }
  const CostGradient<T, P> first = objective(initial);
  if (first.status != Status::SUCCESS) {
    result.status = first.status;
    return result;
  }
  typedef ParameterProblem<T, P> Problem;
  typedef typename Problem::U_Type Vector;
  PythonOptimization::PANOC_Optimizer<Problem, 5> optimizer;
  ObjectiveAdapter<T, P, Objective> adapter(objective);
  // Capture only a pointer. The upstream std::function may allocate at setup
  // on some standard libraries; its numerical workspace is fixed capacity.
  const ObjectiveAdapter<T, P, Objective> *callback = &adapter;
  optimizer.set_cost_function(
      [callback](const Vector &x) { return callback->cost(x); });
  optimizer.set_gradient_function(
      [callback](const Vector &x) { return callback->gradient(x); });
  optimizer.set_tolerance(options.tolerance);
  optimizer.set_max_iteration(options.max_iterations);
  Vector x, lo, hi;
  for (std::size_t j = 0; j < P; ++j) {
    x(j, 0) = initial[j];
    lo(j, 0) = lower[j];
    hi(j, 0) = upper[j];
  }
  optimizer.set_u_min_matrix(lo);
  optimizer.set_u_max_matrix(hi);
  const Vector solution = optimizer.solve(x);
  for (std::size_t j = 0; j < P; ++j)
    result.parameters[j] = solution(j, 0);
  const auto solver_status = optimizer.get_solver_status();
  result.iterations = solver_status.number_of_iteration;
  // Upstream cached cost can describe the previous iterate, so independently
  // evaluate the returned feasible half step and its analytic derivative.
  const CostGradient<T, P> final = objective(result.parameters);
  result.cost = final.cost;
  if (final.status != Status::SUCCESS) {
    result.status = final.status;
    return result;
  }
  for (std::size_t j = 0; j < P; ++j) {
    if (result.parameters[j] < lower[j] || result.parameters[j] > upper[j]) {
      result.status = Status::NON_FINITE;
      return result;
    }
  }
  if (solver_status.exit_status ==
      PythonOptimization::PANOC_ExitStatus::NOT_FINITE_COMPUTATION)
    result.status = Status::NON_FINITE;
  else
    result.status = solver_status.has_converged() ? Status::SUCCESS
                                                  : Status::ITERATION_LIMIT;
  // Verify stationarity at the returned half step, using the actual PANOC
  // step size so its documented ||gamma FPR|| tolerance keeps its meaning.
  const T gamma = optimizer.get_cache().gamma;
  T residual_squared = T(0);
  for (std::size_t j = 0; j < P; ++j) {
    const T trial = result.parameters[j] - gamma * final.gradient[j];
    const T projected =
        trial < lower[j] ? lower[j] : (trial > upper[j] ? upper[j] : trial);
    const T residual = result.parameters[j] - projected;
    residual_squared += residual * residual;
  }
  if (!finite(gamma) || !finite(residual_squared))
    result.status = Status::NON_FINITE;
  else if (result.status == Status::SUCCESS &&
           residual_squared >
               (T(10) * options.tolerance) * (T(10) * options.tolerance))
    result.status = Status::ITERATION_LIMIT;
  return result;
}
} // namespace detail
} // namespace PythonAdaptiveControl
#endif
