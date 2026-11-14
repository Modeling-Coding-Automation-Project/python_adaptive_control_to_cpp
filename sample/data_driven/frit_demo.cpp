// Standalone C++11 counterpart of the Python frit demo.
// No generated header or Python execution is required.
#include "data_driven.hpp"
#include "python_math.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace frit_demo {
using Scalar = double;
// Fixed storage follows the problem dimensions. Change these constants and
// the coefficient definitions in create_design() when changing the problem.
constexpr std::size_t PARAMETER_COUNT = 2;
constexpr std::size_t ORDER = 1;
using Parameters = std::array<Scalar, PARAMETER_COUNT>;
using Filter = PythonAdaptiveControl::RationalFilter<Scalar, ORDER>;
using FilterState = PythonAdaptiveControl::FilterState<Scalar, ORDER>;
using Basis =
    PythonAdaptiveControl::LinearControllerBasis<Scalar, PARAMETER_COUNT,
                                                 ORDER>;
using Fit = PythonAdaptiveControl::FitResult<Scalar, PARAMETER_COUNT>;
using Options = PythonAdaptiveControl::SolverOptions<Scalar>;
using Status = PythonAdaptiveControl::Status;

// The Python design dictionary is a struct. Its array lengths are compile-time
// constants, while the model coefficients and tuning settings are ordinary
// data.
struct Design {
  enum : std::size_t { sample_count = 128, simulation_count = 100 };
  Scalar sample_time;
  Scalar plant_a;
  Scalar plant_b;
  Filter model;
  Basis basis;
  Basis inverse_basis;
  Parameters initial;
  Parameters lower;
  Parameters upper;
  Options options;
};
using Record = std::array<Scalar, Design::sample_count>;
using Trace = std::array<Scalar, Design::simulation_count>;

struct Fits {
  Fit direct;
  Fit iterative;
};

// Named fields replace the Python dictionaries keyed by controller name.
// The same shape holds controllers, filter states, plant states and time
// series.
template <typename Value> struct ControllerValues {
  Value initial;
  Value direct;
  Value iterative;
};
using Controllers = ControllerValues<Filter>;

struct DemoResult {
  Design design;
  Fits fits;
  Controllers controllers;
  Record u;
  Record y;
  Record experiment_time;
  Record experiment_reference;
  Trace time;
  Trace reference;
  Trace desired;
  ControllerValues<Trace> trajectories;
  ControllerValues<Trace> controls;
};

// Runtime error handling belongs to this executable. The library reports Status
// values and does not print messages or throw exceptions.
void require_success(Status status, const char *operation) {
  if (status != Status::SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " +
                             PythonAdaptiveControl::status_name(status));
}

void require_finite(Scalar value, const char *operation) {
  const Scalar largest = std::numeric_limits<Scalar>::max();
  if (!(value >= -largest && value <= largest))
    throw std::runtime_error(std::string(operation) + ": non-finite value");
}

Design create_design() {
  Design design = {};
  design.sample_time = 0.1;
  // M(z) = 0.3*z^-1 / (1 - 0.7*z^-1).
  design.model = Filter({{0.0, 0.3}}, {{1.0, -0.7}});
  // C(z) = Kp + Ki*Ts/(1-z^-1): parameters are [Kp, Ki].
  design.basis =
      Basis({{{{1.0, -1.0}}, {{design.sample_time, 0.0}}}}, {{1.0, -1.0}});
  // The direct method parameterizes C^-1, not the forward PI gains.
  design.inverse_basis = Basis({{{{1.0, 0.0}}, {{0.0, 1.0}}}}, {{1.5, -1.2}});
  // Plant coefficients are used only for data collection and response
  // evaluation.
  design.plant_a = 0.8;
  design.plant_b = 0.2;
  design.initial = {{0.7, 1.0}};
  design.lower = {{0.1, 0.1}};
  design.upper = {{3.0, 6.0}};
  design.options.tolerance = 2e-5;
  design.options.max_iterations = 1000;
  return design;
}

DemoResult run_demo(const Design &design) {
  DemoResult result = {};
  result.design = design;

  // 1. Run a closed-loop unit-step experiment with the fixed initial PI.
  // For a physical system, replace this loop with recorded r_ini, u_ini, y_ini.
  // This controller stays unchanged throughout data collection.
  auto initial_controller = design.basis.controller(design.initial).state();
  Scalar plant_state = 0.0;
  for (std::size_t k = 0; k < Design::sample_count; ++k) {
    result.experiment_time[k] = design.sample_time * Scalar(k);
    result.experiment_reference[k] = 1.0;
    result.y[k] = plant_state;
    require_success(
        initial_controller.step(result.experiment_reference[k] - result.y[k],
                                result.u[k]),
        "initial-controller experiment");
    plant_state = design.plant_a * plant_state + design.plant_b * result.u[k];
    require_finite(plant_state, "experimental plant update");
  }

  // 2. Tune offline with the same recorded data in every optimization step.
  // The tuning API receives u,y, the target model and controller basis; it
  // never receives or simulates the plant used for the experiment.
  const PythonAdaptiveControl::FRITLeastSquares<Scalar, PARAMETER_COUNT, ORDER>
      direct_tuner(design.model, design.inverse_basis);
  const PythonAdaptiveControl::FRIT<Scalar, PARAMETER_COUNT, ORDER> tuner(
      design.model, design.basis);
  result.fits.direct =
      direct_tuner.fit_least_squares(result.u, result.y, design.options);
  result.fits.iterative =
      tuner.fit_iterative(result.u, result.y, design.initial, design.lower,
                          design.upper, design.options);
  require_success(result.fits.direct.status,
                  "FRIT inverse least squares tuning");
  require_success(result.fits.iterative.status, "FRIT PANOC tuning");

  // 3. Construct the initial and both tuned controllers.
  result.controllers.initial = design.basis.controller(design.initial);
  // Convert the fitted inverse controller C^-1 into the executable controller
  // C.
  require_success(design.inverse_basis.controller(result.fits.direct.parameters)
                      .inverse(result.controllers.direct),
                  "FRIT inverse-controller conversion");
  result.controllers.iterative =
      design.basis.controller(result.fits.iterative.parameters);

  // 4. Validate each closed loop independently, with fresh zero initial states.
  ControllerValues<FilterState> controller_states = {
      result.controllers.initial.state(), result.controllers.direct.state(),
      result.controllers.iterative.state()};
  ControllerValues<Scalar> plant_states = {};
  auto model_state = design.model.state();
  for (std::size_t k = 0; k < Design::simulation_count; ++k) {
    result.time[k] = design.sample_time * Scalar(k);
    result.reference[k] = 1.0;
    require_success(model_state.step(result.reference[k], result.desired[k]),
                    "reference model simulation");

    // At time k: y(k)=x(k), u(k)=C(r(k)-y(k)), then x(k+1)=a*x(k)+b*u(k).
    // Using named struct members keeps the Python controller dictionaries
    // visible.
    const auto step_controller = [&](FilterState &controller, Scalar &state,
                                     Trace &output, Trace &control) {
      output[k] = state;
      require_success(
          controller.step(result.reference[k] - output[k], control[k]),
          "closed-loop controller simulation");
      state = design.plant_a * output[k] + design.plant_b * control[k];
      require_finite(state, "validation plant update");
    };
    step_controller(controller_states.initial, plant_states.initial,
                    result.trajectories.initial, result.controls.initial);
    step_controller(controller_states.direct, plant_states.direct,
                    result.trajectories.direct, result.controls.direct);
    step_controller(controller_states.iterative, plant_states.iterative,
                    result.trajectories.iterative, result.controls.iterative);
  }
  return result;
}

void print_fit(const char *name, const Fit &fit) {
  std::cout << name << ": parameters=[";
  for (std::size_t i = 0; i < fit.parameters.size(); ++i)
    std::cout << (i ? ", " : "") << fit.parameters[i];
  std::cout << "] status=" << PythonAdaptiveControl::status_name(fit.status)
            << " cost=" << fit.cost << " iterations=" << fit.iterations << '\n';
}

Scalar tracking_rmse(const Trace &output, const Trace &desired) {
  Scalar scale = 0.0;
  for (std::size_t k = 0; k < output.size(); ++k) {
    const Scalar error = PythonMath::abs(output[k] - desired[k]);
    if (error > scale)
      scale = error;
  }
  if (scale == 0.0)
    return 0.0;
  // Scaling keeps sqrt's input away from the dependency's small-value floor.
  // Zero and near-zero tracking errors are still reported faithfully.
  Scalar mean_square = 0.0;
  for (std::size_t k = 0; k < output.size(); ++k) {
    const Scalar normalized = (output[k] - desired[k]) / scale;
    mean_square += normalized * normalized / Scalar(output.size());
  }
  return scale * PythonMath::sqrt(mean_square);
}

void print_result(const DemoResult &result) {
  const Design &design = result.design;
  std::cout << std::setprecision(17)
            << "FRIT: closed-loop experiment -> offline tuning -> validation\n"
            << "\n[1] Collect data with the fixed initial PI controller\n"
            << "  Kp=" << design.initial[0] << ", Ki=" << design.initial[1]
            << "; unit-step reference, " << Design::sample_count
            << " samples, Ts=" << design.sample_time
            << " s; zero initial states\n"
            << "  Recorded u_ini and y_ini from u_ini = C_ini(r_ini - y_ini).\n"
            << "experiment_time,experiment_reference,experiment_input,"
               "experiment_output\n";
  for (std::size_t k = 0; k < Design::sample_count; ++k)
    std::cout << result.experiment_time[k] << ','
              << result.experiment_reference[k] << ',' << result.u[k] << ','
              << result.y[k] << '\n';

  std::cout << "\n[2] Tune offline using only the same recorded u_ini, y_ini\n"
            << "  r_tilde(rho) = C(rho)^-1 u_ini + y_ini; minimize mean((y_ini "
               "- M r_tilde)^2).\n"
            << "  Direct: inverse-controller coefficients; PANOC: PI gains "
               "[Kp, Ki].\n";
  print_fit("FRIT inverse least squares", result.fits.direct);
  print_fit("FRIT PANOC", result.fits.iterative);

  std::cout << "\n[3] Validate each controller in a fresh closed-loop "
               "unit-step simulation\n"
            << "  " << Design::simulation_count
            << " samples; zero initial states; tracking RMSE relative to M r:\n"
            << "initial PI: RMSE="
            << tracking_rmse(result.trajectories.initial, result.desired)
            << '\n'
            << "FRIT inverse least squares: RMSE="
            << tracking_rmse(result.trajectories.direct, result.desired) << '\n'
            << "FRIT PANOC: RMSE="
            << tracking_rmse(result.trajectories.iterative, result.desired)
            << '\n'
            << "time,reference,desired,y_initial,u_initial,y_direct,u_direct,"
               "y_iterative,u_iterative\n";
  for (std::size_t k = 0; k < Design::simulation_count; ++k)
    std::cout << result.time[k] << ',' << result.reference[k] << ','
              << result.desired[k] << ',' << result.trajectories.initial[k]
              << ',' << result.controls.initial[k] << ','
              << result.trajectories.direct[k] << ','
              << result.controls.direct[k] << ','
              << result.trajectories.iterative[k] << ','
              << result.controls.iterative[k] << '\n';
  std::cout << "FRIT simulation complete.\n";
}
} // namespace frit_demo

int main() {
  try {
    const auto design = frit_demo::create_design();
    const auto result = frit_demo::run_demo(design);
    frit_demo::print_result(result);
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
  }
  return 0;
}
