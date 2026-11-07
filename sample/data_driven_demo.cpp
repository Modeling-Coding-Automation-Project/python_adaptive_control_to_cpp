#include "data_driven.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace PythonAdaptiveControl;
typedef RationalFilter<double, 1> Filter;
typedef LinearControllerBasis<double, 2, 1> Basis;
typedef FitResult<double, 2> Result;
const std::size_t Samples = 256;

// Validation plant only: tuning functions receive the recorded u,y, not G.
// G(z) = .2 z^-1 / (1 - .8 z^-1).
double tracking_mse(const Filter &controller) {
  double y = 0.0, previous_u = 0.0;
  double desired = 0.0, cost = 0.0;
  auto state = controller.state();
  for (std::size_t k = 0; k < 160; ++k) {
    y = 0.8 * y + 0.2 * previous_u;
    const double e = 1.0 - y;
    double u = 0.0;
    if (state.step(e, u) != Status::SUCCESS)
      throw std::runtime_error("Tuned controller execution failed");
    const double difference = y - desired;
    cost += difference * difference / 160.0;
    desired = 0.7 * desired + 0.3;
    previous_u = u;
  }
  return cost;
}

void show(const char *method, const Result &result, const Filter &controller) {
  std::cout << std::left << std::setw(20) << method
            << " parameters=[" << result.parameters[0] << ", "
            << result.parameters[1] << "] status=" << status_name(result.status)
            << " cost=" << result.cost << " iterations=" << result.iterations
            << " step_mse=" << tracking_mse(controller) << '\n';
  if (!result.has_converged() || !std::isfinite(result.cost) ||
      result.cost > 1e-9 || !std::isfinite(tracking_mse(controller)) ||
      tracking_mse(controller) > 1e-8)
    throw std::runtime_error("Tuning or closed-loop validation failed");
}
} // namespace

int main() {
  using namespace PythonAdaptiveControl;
  std::array<double, Samples> u = {}, y = {};
  for (std::size_t k = 0; k < Samples; ++k) {
    const double t = static_cast<double>(k);
    u[k] = std::sin(0.13 * t) + 0.7 * std::cos(0.37 * t)
           + (k % 17 < 8 ? 0.4 : -0.4);
    if (k > 0) y[k] = 0.8 * y[k - 1] + 0.2 * u[k - 1];
  }
  // Desired closed-loop M(z) = .3 z^-1 / (1 - .7 z^-1).
  const Filter model({{0.0, 0.3}}, {{1.0, -0.7}});
  // Backward-Euler PI: C = Kp + Ki * dt/(1-z^-1), dt=.1.
  const Basis forward({{{{1.0, -1.0}}, {{0.1, 0.0}}}}, {{1.0, -1.0}});
  const std::array<double, 2> initial = {{0.6, 1.0}};
  const std::array<double, 2> lower = {{0.1, 0.1}}, upper = {{3.0, 6.0}};
  SolverOptions<double> options;
  options.tolerance = 1e-10;
  options.max_iterations = 1000;

  std::cout << std::setprecision(10)
            << "Expected PI gains: [Kp, Ki] = [1.2, 3]\n"
            << "Initial PI step MSE: " << tracking_mse(forward.controller(initial)) << '\n';
  const Result vrft_ls = vrft_least_squares(u, y, model, forward, options);
  show("VRFT QR", vrft_ls, forward.controller(vrft_ls.parameters));
  const Result vrft_panoc = vrft_optimize(u, y, model, forward, initial, lower, upper, options);
  show("VRFT PANOC", vrft_panoc, forward.controller(vrft_panoc.parameters));
  const Result frit_panoc = frit_optimize(u, y, model, forward, initial, lower, upper, options);
  show("FRIT PANOC", frit_panoc, forward.controller(frit_panoc.parameters));

  // Exact noniterative FRIT uses an INVERSE controller family:
  // C^-1 = (rho0 + rho1 z^-1)/(1.5 - 1.2 z^-1).
  // Expected inverse coefficients [1,-1] are NOT [Kp,Ki].
  const Basis inverse_basis({{{{1.0, 0.0}}, {{0.0, 1.0}}}}, {{1.5, -1.2}});
  const Result frit_ls = frit_least_squares(u, y, model, inverse_basis, options);
  Filter tuned_controller;
  if (inverse_basis.controller(frit_ls.parameters).inverse(tuned_controller) != Status::SUCCESS)
    throw std::runtime_error("The fitted inverse controller cannot be realized causally");
  show("FRIT inverse QR", frit_ls, tuned_controller);
  return 0;
}
