// Run sample/vrft_demo.py to regenerate this Python-designed configuration.
#include "generated/vrft_demo.hpp"
#include "data_driven_sample_common.hpp"

int main() {
  const auto u = vrft_demo::recorded_input();
  const auto y = vrft_demo::recorded_output();
  const auto options = vrft_demo::make_options();
  const auto tuner = vrft_demo::make_tuner();
  const auto direct = tuner.fit_least_squares(u, y, options);
  const auto iterative = tuner.fit_iterative(
      u, y, vrft_demo::initial_parameters(), vrft_demo::lower_bounds(),
      vrft_demo::upper_bounds(), options);
  const auto basis = vrft_demo::make_forward_basis();
  const auto model = vrft_demo::make_model();
  const bool direct_ok = DataDrivenSample::show(
      "VRFT least squares", direct, vrft_demo::least_squares_parameters(),
      basis.controller(direct.parameters), model);
  const bool iterative_ok = DataDrivenSample::show(
      "VRFT PANOC", iterative, vrft_demo::optimized_parameters(),
      basis.controller(iterative.parameters), model);

  // Deploy an offline-tuned controller without rerunning the tuning algorithm.
  const double deployed_error = DataDrivenSample::step_mse(
      vrft_demo::make_optimized_controller(), model);
  std::cout << "Python-generated controller step MSE: " << deployed_error << '\n';
  return direct_ok && iterative_ok &&
                 PythonAdaptiveControl::detail::finite(deployed_error) && deployed_error < 1e-7
             ? 0 : 1;
}
