#ifndef DATA_DRIVEN_SAMPLE_COMMON_HPP_
#define DATA_DRIVEN_SAMPLE_COMMON_HPP_

#include "data_driven.hpp"
#include "python_math.hpp"

#include <iomanip>
#include <iostream>
#include <limits>

namespace DataDrivenSample {
// The known plant is used only to validate a controller on a new step record.
// Tuning functions receive the recorded u,y and never receive this plant.
template <typename Filter>
double step_mse(const Filter &controller, const Filter &model) {
  auto controller_state = controller.state();
  auto model_state = model.state();
  double output = 0.0, previous_input = 0.0, mse = 0.0;
  for (std::size_t k = 0; k < 100; ++k) {
    output = 0.8 * output + 0.2 * previous_input;
    double desired = 0.0, control = 0.0;
    if (model_state.step(1.0, desired) != PythonAdaptiveControl::Status::SUCCESS ||
        controller_state.step(1.0 - output, control) != PythonAdaptiveControl::Status::SUCCESS)
      return std::numeric_limits<double>::infinity();
    const double difference = output - desired;
    mse += difference * difference / 100.0;
    previous_input = control;
  }
  return mse;
}

template <typename Result, typename Parameters, typename Filter>
bool show(const char *name, const Result &fit, const Parameters &python_parameters,
          const Filter &controller, const Filter &model) {
  const double mse = step_mse(controller, model);
  std::cout << std::setprecision(10) << name << ": parameters=[";
  for (std::size_t index = 0; index < fit.parameters.size(); ++index)
    std::cout << (index ? ", " : "") << fit.parameters[index];
  std::cout << "] status=" << PythonAdaptiveControl::status_name(fit.status)
            << " cost=" << fit.cost << " iterations=" << fit.iterations
            << " step_mse=" << mse << '\n';
  if (!fit.has_converged() || !PythonAdaptiveControl::detail::finite(fit.cost) ||
      !PythonAdaptiveControl::detail::finite(mse) || mse > 1e-7)
    return false;
  for (std::size_t index = 0; index < fit.parameters.size(); ++index) {
    if (PythonMath::abs(fit.parameters[index] - python_parameters[index]) > 1e-5)
      return false;
  }
  return true;
}
} // namespace DataDrivenSample
#endif
