// Run sample/frit_demo.py to regenerate this Python-designed configuration.
#include "generated/frit_demo.hpp"
#include "data_driven_sample_common.hpp"

int main() {
  const auto u = frit_demo::recorded_input();
  const auto y = frit_demo::recorded_output();
  const auto options = frit_demo::make_options();
  const auto direct =
      frit_demo::make_inverse_tuner().fit_least_squares(u, y, options);
  const auto iterative = frit_demo::make_tuner().fit_iterative(
      u, y, frit_demo::initial_parameters(), frit_demo::lower_bounds(),
      frit_demo::upper_bounds(), options);
  const auto model = frit_demo::make_model();

  // Direct FRIT identifies coefficients of C^-1, not forward PI gains.
  // Invert the identified transfer function before executing the controller.
  frit_demo::Filter direct_controller;
  if (frit_demo::make_inverse_basis()
          .controller(direct.parameters)
          .inverse(direct_controller) != PythonAdaptiveControl::Status::SUCCESS)
    return 1;
  const bool direct_ok = DataDrivenSample::show(
      "FRIT inverse least squares", direct,
      frit_demo::least_squares_parameters(), direct_controller, model);
  const bool iterative_ok = DataDrivenSample::show(
      "FRIT PANOC", iterative, frit_demo::optimized_parameters(),
      frit_demo::make_forward_basis().controller(iterative.parameters), model);

  // The generator has already converted the inverse-linear fit into C.
  const double deployed_error = DataDrivenSample::step_mse(
      frit_demo::make_least_squares_controller(), model);
  std::cout << "Python-generated controller step MSE: " << deployed_error
            << '\n';
  return direct_ok && iterative_ok &&
                 PythonAdaptiveControl::detail::finite(deployed_error) &&
                 deployed_error < 1e-7
             ? 0
             : 1;
}
