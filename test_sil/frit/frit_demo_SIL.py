"""FRIT SIL: execute the sample workflow with Python and C++ library objects.

Run ``python test_sil/frit/frit_demo_SIL.py``. SIL_Builder compiles the paired
frit_SIL.cpp binding; no sample executable, generated controller, or plot is used.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from external_libraries.MCAP_python_adaptive_control.data_driven import (
    FRIT, FRITLeastSquares, LinearControllerBasis, RationalFilter, SolverOptions,
)
from test_sil.SIL_operator import SIL_Builder
from test_sil.SIL_test_utils import (
    compare_fit, expect_near, experiment, independent_filter,
    make_cpp_options, regression_cases,
)
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester


def create_design():
    """The same model, PI basis and settings as sample/data_driven/frit_demo."""
    sample_time = 0.1
    return {
        "sample_time": sample_time,
        "sample_count": 128,
        "simulation_count": 100,
        "plant_a": 0.8,
        "plant_b": 0.2,
        "model": RationalFilter([0.0, 0.3], [1.0, -0.7]),
        "basis": LinearControllerBasis.pi(sample_time),
        # Direct FRIT fits C^-1; these parameters are not forward PI gains.
        "inverse_basis": LinearControllerBasis(
            [[1.0, 0.0], [0.0, 1.0]], [1.5, -1.2]),
        "initial": np.array([0.7, 1.0]),
        "lower": np.array([0.1, 0.1]),
        "upper": np.array([3.0, 6.0]),
        "options": SolverOptions(tolerance=2e-5, max_iterations=1000),
    }


def run_sample_workflow(cpp, tester):
    design = create_design()
    model, basis = design["model"], design["basis"]
    inverse_basis, options = design["inverse_basis"], design["options"]
    model_cpp = cpp.RationalFilter(model.numerator, model.denominator)
    basis_cpp = cpp.LinearControllerBasis(basis.numerators, basis.denominator)
    inverse_basis_cpp = cpp.LinearControllerBasis(
        inverse_basis.numerators, inverse_basis.denominator)
    options_cpp = make_cpp_options(cpp, options)

    # 1. Collect a unit-step record with the fixed initial PI controller.
    # Each implementation has its own controller state and simulated plant.
    reference = np.ones(design["sample_count"])
    u, y = np.zeros(reference.size), np.zeros(reference.size)
    u_cpp, y_cpp = np.zeros(reference.size), np.zeros(reference.size)
    initial_state = basis.controller(design["initial"]).state()
    initial_state_cpp = basis_cpp.controller(design["initial"]).state()
    plant_state = plant_state_cpp = 0.0
    for k, command in enumerate(reference):
        y[k], y_cpp[k] = plant_state, plant_state_cpp
        u[k] = initial_state.step(command - y[k])
        u_cpp[k] = initial_state_cpp.step(command - y_cpp[k])
        expect_near(tester, y_cpp[k], y[k], 1e-12, f"FRIT experiment y[{k}]")
        expect_near(tester, u_cpp[k], u[k], 1e-12, f"FRIT experiment u[{k}]")
        plant_state = design["plant_a"] * y[k] + design["plant_b"] * u[k]
        plant_state_cpp = (design["plant_a"] * y_cpp[k]
                           + design["plant_b"] * u_cpp[k])
    expect_near(tester, [u_cpp[0], y_cpp[0], y_cpp[1]], [0.8, 0.0, 0.16],
                1e-12, "FRIT initial PI experiment timing")

    # 2. Reuse this fixed record in both offline tuning implementations.
    # No plant model is supplied to either tuning API.
    direct_tuner = FRITLeastSquares(model, inverse_basis)
    tuner = FRIT(model, basis)
    direct_tuner_cpp = cpp.FRITLeastSquares(model_cpp, inverse_basis_cpp)
    tuner_cpp = cpp.FRIT(model_cpp, basis_cpp)
    direct = direct_tuner.fit_least_squares(u, y, options)
    direct_cpp = direct_tuner_cpp.fit_least_squares(u, y, options_cpp)
    iterative = tuner.fit_iterative(
        u, y, design["initial"], design["lower"], design["upper"], options)
    iterative_cpp = tuner_cpp.fit_iterative(
        u, y, design["initial"], design["lower"], design["upper"], options_cpp)
    compare_fit(tester, direct_cpp, direct, options,
                "FRIT inverse least squares")
    compare_fit(tester, iterative_cpp, iterative, options, "FRIT PANOC")
    matrix, target = direct_tuner.regression(u, y, options)
    matrix_cpp, target_cpp = direct_tuner_cpp.regression(u, y, options_cpp)
    expect_near(tester, matrix_cpp, matrix, 3e-11,
                "FRIT sample regression matrix")
    expect_near(tester, target_cpp, target, 3e-11,
                "FRIT sample regression target")
    cost, gradient = tuner.cost_gradient(design["initial"], u, y, options)
    cost_cpp, gradient_cpp = tuner_cpp.cost_gradient(
        design["initial"], u, y, options_cpp)
    expect_near(tester, cost_cpp, cost, 2e-10, "FRIT sample initial cost")
    expect_near(tester, gradient_cpp, gradient,
                2e-10, "FRIT sample initial gradient")

    # 3. Construct the controllers, explicitly converting C^-1 to C.
    controllers = {
        "initial PI": basis.controller(design["initial"]),
        "FRIT inverse least squares": inverse_basis.controller(direct.parameters).inverse(),
        "FRIT PANOC": basis.controller(iterative.parameters),
    }
    controllers_cpp = {
        "initial PI": basis_cpp.controller(design["initial"]),
        "FRIT inverse least squares": inverse_basis_cpp.controller(direct_cpp.parameters).inverse(),
        "FRIT PANOC": basis_cpp.controller(iterative_cpp.parameters),
    }
    # Compare the offline signals shown on the Python sample's second page.
    for name, controller in controllers.items():
        fictitious_reference = controller.inverse().filter(u) + y
        fictitious_reference_cpp = controllers_cpp[name].inverse().filter(
            u) + y
        expect_near(tester, fictitious_reference_cpp, fictitious_reference, 1e-6,
                    "FRIT fictitious reference: " + name)
        expect_near(tester, model_cpp.filter(fictitious_reference_cpp),
                    model.filter(fictitious_reference), 1e-6,
                    "FRIT offline model response: " + name)

    # 4. Reset all states, then compare every sample of three independent loops.
    states = {name: controller.state()
              for name, controller in controllers.items()}
    states_cpp = {name: controller.state()
                  for name, controller in controllers_cpp.items()}
    plant_states = {name: 0.0 for name in controllers}
    plant_states_cpp = {name: 0.0 for name in controllers}
    trajectories = {name: np.zeros(
        design["simulation_count"]) for name in controllers}
    trajectories_cpp = {name: np.zeros(
        design["simulation_count"]) for name in controllers}
    controls_cpp = {name: np.zeros(
        design["simulation_count"]) for name in controllers}
    desired = np.zeros(design["simulation_count"])
    desired_cpp = np.zeros_like(desired)
    model_state, model_state_cpp = model.state(), model_cpp.state()
    for k in range(design["simulation_count"]):
        command = 1.0
        desired[k] = model_state.step(command)
        desired_cpp[k] = model_state_cpp.step(command)
        expect_near(tester, desired_cpp[k], desired[k], 1e-12,
                    f"FRIT validation target[{k}]")
        for name in controllers:
            output, output_cpp = plant_states[name], plant_states_cpp[name]
            control = states[name].step(command - output)
            control_cpp = states_cpp[name].step(command - output_cpp)
            expect_near(tester, output_cpp, output, 1e-7, f"{name} y[{k}]")
            expect_near(tester, control_cpp, control, 1e-7, f"{name} u[{k}]")
            trajectories[name][k], trajectories_cpp[name][k] = output, output_cpp
            controls_cpp[name][k] = control_cpp
            plant_states[name] = design["plant_a"] * \
                output + design["plant_b"] * control
            plant_states_cpp[name] = (design["plant_a"] * output_cpp
                                      + design["plant_b"] * control_cpp)
    direct_name = "FRIT inverse least squares"
    expect_near(tester, [trajectories_cpp[direct_name][0], controls_cpp[direct_name][0],
                         trajectories_cpp[direct_name][1]], [0.0, 1.5, 0.3],
                1e-10, "FRIT fresh validation state and one-sample delay")
    for name in controllers:
        rmse = np.sqrt(np.mean((trajectories[name] - desired) ** 2))
        rmse_cpp = np.sqrt(
            np.mean((trajectories_cpp[name] - desired_cpp) ** 2))
        expect_near(tester, rmse_cpp, rmse, 1e-7, name + " tracking RMSE")
    tester.throw_error_if_test_failed()
    print("PASS FRIT sample: fixed-PI acquisition, offline fitting, inverse conversion, "
          "and fresh closed-loop responses")


def run_regression_checks(cpp, tester):
    """Retain the exact, noisy/ridge/discard, and two-sample-delay SIL cases."""
    basis = LinearControllerBasis.pi(0.1)
    inverse_basis = LinearControllerBasis(
        [[1.0, 0.0], [0.0, 1.0]], [1.0, -0.7])
    basis_cpp = cpp.LinearControllerBasis(basis.numerators, basis.denominator)
    inverse_basis_cpp = cpp.LinearControllerBasis(
        inverse_basis.numerators, inverse_basis.denominator)
    initial, lower, upper = [0.9, 1.4], [0.1, 0.1], [2.0, 5.0]
    for name, b, a, noise, ridge, discard in regression_cases():
        u, y = experiment(noise)
        model = RationalFilter(b, a)
        model_cpp = cpp.RationalFilter(b, a)
        options = SolverOptions(ridge=ridge, tolerance=2e-5,
                                max_iterations=1000, discard_samples=discard)
        options_cpp = make_cpp_options(cpp, options)
        direct_tuner = FRITLeastSquares(model, inverse_basis)
        direct_tuner_cpp = cpp.FRITLeastSquares(model_cpp, inverse_basis_cpp)
        tuner, tuner_cpp = FRIT(model, basis), cpp.FRIT(model_cpp, basis_cpp)
        direct = direct_tuner.fit_least_squares(u, y, options)
        direct_cpp = direct_tuner_cpp.fit_least_squares(u, y, options_cpp)
        iterative = tuner.fit_iterative(u, y, initial, lower, upper, options)
        iterative_cpp = tuner_cpp.fit_iterative(
            u, y, initial, lower, upper, options_cpp)
        compare_fit(tester, direct_cpp, direct, options,
                    name + "/FRIT least squares")
        compare_fit(tester, iterative_cpp, iterative,
                    options, name + "/FRIT PANOC")

        matrix, target = direct_tuner.regression(u, y, options)
        matrix_cpp, target_cpp = direct_tuner_cpp.regression(u, y, options_cpp)
        expect_near(tester, matrix_cpp, matrix,
                    3e-10, name + "/FRIT regressors")
        expect_near(tester, target_cpp, target, 3e-10, name + "/FRIT target")
        inverse_columns = np.column_stack([
            independent_filter(u, [1.0, 0.0], [1.0, -0.7]),
            independent_filter(u, [0.0, 1.0], [1.0, -0.7]),
        ])
        independent_matrix = np.column_stack([
            independent_filter(column, b, a) for column in inverse_columns.T])
        independent_target = y - independent_filter(y, b, a)
        expect_near(tester, matrix_cpp, independent_matrix[discard:], 3e-10,
                    name + "/independent FRIT regressors")
        expect_near(tester, target_cpp, independent_target[discard:], 3e-10,
                    name + "/independent FRIT target")
        cost, gradient = tuner.cost_gradient(initial, u, y, options)
        cost_cpp, gradient_cpp = tuner_cpp.cost_gradient(
            initial, u, y, options_cpp)
        expect_near(tester, cost_cpp, cost, 2e-10, name + "/FRIT cost")
        expect_near(tester, gradient_cpp, gradient,
                    2e-10, name + "/FRIT gradient")

        # Validate the tuned PI on a fresh record for the .7/.3 fixture plant.
        state = basis.controller(iterative.parameters).state()
        state_cpp = basis_cpp.controller(iterative_cpp.parameters).state()
        plant_state = plant_state_cpp = 0.0
        for k in range(64):
            expect_near(tester, plant_state_cpp, plant_state, 6e-5,
                        f"{name}/FRIT held-out y[{k}]")
            control = state.step(1.0 - plant_state)
            control_cpp = state_cpp.step(1.0 - plant_state_cpp)
            expect_near(tester, control_cpp, control, 6e-5,
                        f"{name}/FRIT held-out u[{k}]")
            plant_state = 0.7 * plant_state + 0.3 * control
            plant_state_cpp = 0.7 * plant_state_cpp + 0.3 * control_cpp
        if name == "exact":
            expect_near(tester, direct_cpp.parameters, [1.2, -1.2], 1e-7,
                        "FRIT analytical inverse coefficients")
            expect_near(tester, iterative_cpp.parameters, [7.0 / 12.0, 2.5], 2e-4,
                        "FRIT analytical forward PI gains")
        tester.throw_error_if_test_failed()
        print("PASS FRIT " + name +
              ": fits, regressors, objective, gradient, and response")


def main():
    tester = MCAPTester()
    with SIL_Builder(Path(__file__).parent) as cpp:
        run_sample_workflow(cpp, tester)
        run_regression_checks(cpp, tester)
    tester.throw_error_if_test_failed()
    print("FRIT SIL passed")


if __name__ == "__main__":
    main()
