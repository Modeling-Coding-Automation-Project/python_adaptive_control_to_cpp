"""VRFT SIL: compare Python/C++ library calls along the sample design workflow.

Run with ``python test_sil/vrft/vrft_demo_SIL.py``. The paired C++ wrapper is
built and imported with pybind11; no sample executable or plot is used.
"""
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from external_libraries.MCAP_python_adaptive_control.data_driven import (
    LinearControllerBasis, RationalFilter, SolverOptions, VRFT,
)
from test_sil.SIL_operator import SIL_Builder
from test_sil.SIL_test_utils import (
    compare_fit, expect_near, experiment, independent_filter,
    make_cpp_options, regression_cases,
)
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester


NEAR_LIMIT = 1e-5


def run_sample_workflow(cpp, tester):
    # Use the same design as sample/data_driven/vrft_demo.cpp and the Python demo.
    sample_time = 0.1
    sample_count, simulation_count = 128, 100
    plant_a, plant_b = 0.8, 0.2
    model = RationalFilter([0.0, 0.3], [1.0, -0.7])
    basis = LinearControllerBasis.pi(sample_time)
    initial = np.array([0.7, 1.0])
    lower, upper = np.array([0.1, 0.1]), np.array([3.0, 6.0])
    options = SolverOptions(tolerance=2e-5, max_iterations=1000)

    model_cpp = cpp.RationalFilter(model.numerator, model.denominator)
    basis_cpp = cpp.LinearControllerBasis(basis.numerators, basis.denominator)
    options_cpp = make_cpp_options(cpp, options)

    # 1. Acquire data with the fixed initial PI in two independent closed loops.
    # Compare each C++ controller update with the Python library update.
    reference = np.ones(sample_count)
    u, y = np.zeros(sample_count), np.zeros(sample_count)
    u_cpp, y_cpp = np.zeros(sample_count), np.zeros(sample_count)
    controller = basis.controller(initial).state()
    controller_cpp = basis_cpp.controller(initial).state()
    plant_state = plant_state_cpp = 0.0
    for k, command in enumerate(reference):
        y[k], y_cpp[k] = plant_state, plant_state_cpp
        u[k] = controller.step(command - y[k])
        u_cpp[k] = controller_cpp.step(command - y_cpp[k])
        expect_near(tester, y_cpp[k], y[k], 1e-12, f"VRFT acquisition y[{k}]")
        expect_near(tester, u_cpp[k], u[k], 1e-12, f"VRFT acquisition u[{k}]")
        plant_state = plant_a * plant_state + plant_b * u[k]
        plant_state_cpp = plant_a * plant_state_cpp + plant_b * u_cpp[k]
    expect_near(tester, u_cpp[0], 0.8, 1e-12, "VRFT initial PI input at k=0")
    expect_near(tester, y_cpp[0], 0.0, 0.0, "VRFT experiment starts at zero")
    expect_near(tester, y_cpp[1], 0.16, 1e-12, "VRFT one-sample plant delay")

    # 2. Tune offline: both libraries receive the same fixed recorded u,y.
    tuner = VRFT(model, basis)
    tuner_cpp = cpp.VRFT(model_cpp, basis_cpp)
    recorded_u, recorded_y = u.copy(), y.copy()
    direct = tuner.fit_least_squares(u, y, options)
    direct_cpp = tuner_cpp.fit_least_squares(u, y, options_cpp)
    iterative = tuner.fit_iterative(u, y, initial, lower, upper, options)
    iterative_cpp = tuner_cpp.fit_iterative(
        u, y, initial, lower, upper, options_cpp)
    compare_fit(tester, direct_cpp, direct, options,
                "VRFT sample least squares")
    compare_fit(tester, iterative_cpp, iterative, options, "VRFT sample PANOC")
    expect_near(tester, u, recorded_u, 0.0,
                "VRFT recorded input remains fixed")
    expect_near(tester, y, recorded_y, 0.0,
                "VRFT recorded output remains fixed")
    columns, target = tuner.regression(u, y, options)
    columns_cpp, target_cpp = tuner_cpp.regression(u, y, options_cpp)
    expect_near(tester, columns_cpp, columns, 1e-10, "VRFT sample regressors")
    expect_near(tester, target_cpp, target, 1e-12,
                "VRFT sample regression target")

    # 3. Build initial/direct/iterative controllers using each library's results.
    controllers = {
        "initial PI": basis.controller(initial),
        "VRFT least squares": basis.controller(direct.parameters),
        "VRFT PANOC": basis.controller(iterative.parameters),
    }
    controllers_cpp = {
        "initial PI": basis_cpp.controller(initial),
        "VRFT least squares": basis_cpp.controller(direct_cpp.parameters),
        "VRFT PANOC": basis_cpp.controller(iterative_cpp.parameters),
    }

    # Compare the offline signals shown on the Python demo's second plot page.
    # With L=M, the fitted signals are M u_ini and C(rho)(I-M)y_ini.
    error_filtered = y - model.filter(y)
    error_filtered_cpp = y - model_cpp.filter(y)
    expect_near(tester, model_cpp.filter(u), model.filter(u), 1e-12,
                "VRFT offline filtered input M u_ini")
    for name, controller in controllers.items():
        expect_near(tester, controllers_cpp[name].filter(error_filtered_cpp),
                    controller.filter(error_filtered), NEAR_LIMIT,
                    f"VRFT offline C(I-M)y_ini: {name}")

    # 4. Validate in fresh, independent closed loops; preserve state each step.
    states = {name: controller.state()
              for name, controller in controllers.items()}
    states_cpp = {name: controller.state()
                  for name, controller in controllers_cpp.items()}
    plants = {name: 0.0 for name in controllers}
    plants_cpp = {name: 0.0 for name in controllers}
    outputs = {name: np.zeros(simulation_count) for name in controllers}
    outputs_cpp = {name: np.zeros(simulation_count) for name in controllers}
    desired, desired_cpp = np.zeros(
        simulation_count), np.zeros(simulation_count)
    model_state, model_state_cpp = model.state(), model_cpp.state()
    for k in range(simulation_count):
        command = 1.0
        desired[k], desired_cpp[k] = model_state.step(
            command), model_state_cpp.step(command)
        expect_near(tester, desired_cpp[k], desired[k],
                    1e-12, f"VRFT target output[{k}]")
        for name in controllers:
            output, output_cpp = plants[name], plants_cpp[name]
            control = states[name].step(command - output)
            control_cpp = states_cpp[name].step(command - output_cpp)
            outputs[name][k], outputs_cpp[name][k] = output, output_cpp
            expect_near(tester, output_cpp, output, NEAR_LIMIT,
                        f"{name} validation y[{k}]")
            expect_near(tester, control_cpp, control,
                        NEAR_LIMIT, f"{name} validation u[{k}]")
            plants[name] = plant_a * output + plant_b * control
            plants_cpp[name] = plant_a * output_cpp + plant_b * control_cpp
            if k == 0:
                expect_near(tester, output_cpp, 0.0, 0.0,
                            f"{name} validation resets state")
                if name == "VRFT least squares":
                    expect_near(tester, control_cpp, 1.5,
                                1e-10, "VRFT tuned first input")
    expect_near(tester, outputs_cpp["VRFT least squares"][1], 0.3, 1e-10,
                "VRFT tuned response has one sample of delay")
    for name in controllers:
        rmse = np.sqrt(np.mean((outputs[name] - desired) ** 2))
        rmse_cpp = np.sqrt(np.mean((outputs_cpp[name] - desired_cpp) ** 2))
        expect_near(tester, rmse_cpp, rmse, 1e-7, f"{name} tracking RMSE")
    tester.throw_error_if_test_failed()
    print("PASS VRFT sample: acquisition, offline least squares/PANOC, filter signals, validation")


def run_regression_checks(cpp, tester):
    """Retain the exact/noisy/regularized/delayed cases of the earlier SIL."""
    basis = LinearControllerBasis.pi(0.1)
    basis_cpp = cpp.LinearControllerBasis(basis.numerators, basis.denominator)
    initial, lower, upper = [0.9, 1.4], [0.1, 0.1], [2.0, 5.0]
    for name, numerator, denominator, noise, ridge, discard in regression_cases():
        u, y = experiment(noise)
        model = RationalFilter(numerator, denominator)
        model_cpp = cpp.RationalFilter(numerator, denominator)
        options = SolverOptions(ridge=ridge, tolerance=2e-5, max_iterations=1000,
                                discard_samples=discard)
        options_cpp = make_cpp_options(cpp, options)
        tuner, tuner_cpp = VRFT(model, basis), cpp.VRFT(model_cpp, basis_cpp)
        direct = tuner.fit_least_squares(u, y, options)
        direct_cpp = tuner_cpp.fit_least_squares(u, y, options_cpp)
        iterative = tuner.fit_iterative(u, y, initial, lower, upper, options)
        iterative_cpp = tuner_cpp.fit_iterative(
            u, y, initial, lower, upper, options_cpp)
        compare_fit(tester, direct_cpp, direct, options,
                    f"VRFT {name} least squares")
        compare_fit(tester, iterative_cpp, iterative,
                    options, f"VRFT {name} PANOC")
        columns, target = tuner.regression(u, y, options)
        columns_cpp, target_cpp = tuner_cpp.regression(u, y, options_cpp)
        expect_near(tester, columns_cpp, columns,
                    1e-10, f"VRFT {name} regressors")
        expect_near(tester, target_cpp, target, 1e-10, f"VRFT {name} target")

        # These direct-form recurrences are independent of both filter classes.
        error = y - independent_filter(y, numerator, denominator)
        independent_columns = np.column_stack(
            (error, 0.1 * np.cumsum(error)))[discard:]
        independent_target = independent_filter(
            u, numerator, denominator)[discard:]
        expect_near(tester, columns_cpp, independent_columns, 1e-10,
                    f"VRFT {name} independent regressors")
        expect_near(tester, target_cpp, independent_target, 1e-10,
                    f"VRFT {name} independent target")
        for fit_name, fit, fit_cpp in (("least squares", direct, direct_cpp),
                                       ("PANOC", iterative, iterative_cpp)):
            parameters = np.asarray(fit_cpp.parameters)
            residual = independent_target - independent_columns @ parameters
            objective = np.mean(residual ** 2) + ridge * \
                (parameters @ parameters)
            expect_near(tester, fit_cpp.cost, objective, 2e-9,
                        f"VRFT {name}/{fit_name} independent objective")
            state = basis.controller(fit.parameters).state()
            state_cpp = basis_cpp.controller(fit_cpp.parameters).state()
            output = output_cpp = 0.0
            response_cpp = np.zeros(64)
            for k in range(response_cpp.size):
                response_cpp[k] = output_cpp
                control = state.step(1.0 - output)
                control_cpp = state_cpp.step(1.0 - output_cpp)
                expect_near(tester, output_cpp, output, NEAR_LIMIT,
                            f"VRFT {name}/{fit_name} held-out output[{k}]")
                expect_near(tester, control_cpp, control, NEAR_LIMIT,
                            f"VRFT {name}/{fit_name} held-out input[{k}]")
                output = 0.7 * output + 0.3 * control
                output_cpp = 0.7 * output_cpp + 0.3 * control_cpp
            if name == "exact":
                expect_near(tester, parameters, [7.0 / 12.0, 2.5], 2e-4,
                            f"VRFT analytical {fit_name} parameters")
                if fit_name == "least squares":
                    expect_near(tester, response_cpp, 1.0 - 0.75 ** np.arange(64),
                                1e-10, "VRFT analytical held-out step response")
        tester.throw_error_if_test_failed()
        print(
            f"PASS VRFT {name}: fits, regressors, independent objective, held-out responses")


def main():
    tester = MCAPTester()
    with SIL_Builder(Path(__file__).parent) as cpp:
        run_sample_workflow(cpp, tester)
        run_regression_checks(cpp, tester)
    tester.throw_error_if_test_failed()
    print("VRFT SIL passed.")


if __name__ == "__main__":
    main()
