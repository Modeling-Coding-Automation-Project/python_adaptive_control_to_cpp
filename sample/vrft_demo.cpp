// Run sample/vrft_demo.py first to generate the problem types and settings.
#include "vrft_demo.hpp"

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

template <typename Result> void print_fit(const char *name, const Result &fit) {
  std::cout << name << ": parameters=[";
  for (std::size_t i = 0; i < fit.parameters.size(); ++i)
    std::cout << (i ? ", " : "") << fit.parameters[i];
  std::cout << "] status=" << PythonAdaptiveControl::status_name(fit.status)
            << " cost=" << fit.cost << " iterations=" << fit.iterations << '\n';
}

int main() {
  // 1. Use the same problem configuration as create_design() in the Python
  // demo. Sizes, filter order, model and controller basis depend on the design
  // problem.
  using namespace vrft_demo;
  using PythonAdaptiveControl::Status;
  const auto model = make_model();
  const auto basis = make_forward_basis();
  const auto options = make_options();
  const auto initial = initial_parameters();
  const auto lower = lower_bounds();
  const auto upper = upper_bounds();
  std::cout << std::setprecision(15);

  // 2. Collect one input/output experiment from a simulated physical plant.
  // x(k+1) = a*x(k) + b*u(k), y(k) = x(k): one sample of input/output delay.
  // Replace this experiment with measured u,y when using a real plant.
  Record u = {}, y = {};
  Scalar experiment_state = 0.0;
  std::uint32_t random_state = EXCITATION_SEED;
  std::cout << "experiment_time,experiment_input,experiment_output\n";
  for (std::size_t k = 0; k < SAMPLE_COUNT; ++k) {
    random_state = 1664525u * random_state + 1013904223u;
    u[k] = Scalar(random_state >> 16) / Scalar(32768.0) - Scalar(1.0);
    y[k] = experiment_state;
    std::cout << Scalar(k) * SAMPLE_TIME << ',' << u[k] << ',' << y[k] << '\n';
    experiment_state = PLANT_A * experiment_state + PLANT_B * u[k];
  }

  // 3. Tune the controller using only the measured record, target model and
  // basis.
  const auto tuner = make_tuner();
  const auto direct = tuner.fit_least_squares(u, y, options);
  const auto iterative =
      tuner.fit_iterative(u, y, initial, lower, upper, options);
  print_fit("VRFT least squares", direct);
  print_fit("VRFT PANOC", iterative);
  if (!direct.has_converged() || !iterative.has_converged()) {
    std::cout << "ERROR: controller tuning did not converge.\n";
    return 0;
  }

  // 4. Realize the initial and both newly tuned controllers.
  const auto initial_controller = basis.controller(initial);
  const auto direct_controller = basis.controller(direct.parameters);
  const auto iterative_controller = basis.controller(iterative.parameters);
  const std::array<Filter, 3> controllers = {
      {initial_controller, direct_controller, iterative_controller}};
  std::array<decltype(initial_controller.state()), 3> controller_states = {
      {controllers[0].state(), controllers[1].state(), controllers[2].state()}};

  // 5. Run independent closed loops with the same unit-step command.
  // Keep each controller and plant state across samples; reset them only here.
  auto model_state = model.state();
  std::array<Scalar, 3> plant_states = {};
  std::cout << "time,reference,desired,y_initial,u_initial,y_direct,u_direct,"
               "y_iterative,u_iterative\n";
  for (std::size_t k = 0; k < SIMULATION_COUNT; ++k) {
    const Scalar reference = 1.0;
    Scalar desired = 0.0;
    if (model_state.step(reference, desired) != Status::SUCCESS) {
      std::cout << "ERROR: target model simulation failed.\n";
      return 0;
    }
    const auto measured_outputs = plant_states; // y(k) = x(k)
    std::array<Scalar, 3> controls = {};
    for (std::size_t i = 0; i < controller_states.size(); ++i) {
      const Scalar error = reference - measured_outputs[i];
      if (controller_states[i].step(error, controls[i]) != Status::SUCCESS) {
        std::cout << "ERROR: controller simulation failed at sample " << k
                  << '\n';
        return 0;
      }
      plant_states[i] = PLANT_A * measured_outputs[i] + PLANT_B * controls[i];
    }
    // Python plots these same signals with SimulationPlotterDash.
    // C++ shows the sample time, reference, target output and each (y,u) pair.
    std::cout << Scalar(k) * SAMPLE_TIME << ',' << reference << ',' << desired;
    for (std::size_t i = 0; i < controller_states.size(); ++i)
      std::cout << ',' << measured_outputs[i] << ',' << controls[i];
    std::cout << '\n';
  }
  std::cout << "VRFT simulation complete.\n";
  return 0;
}
