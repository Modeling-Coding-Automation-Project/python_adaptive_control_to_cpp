"""Compare NumPy and C++11 implementations on independently generated records.

Run after building the CMake target ``data_driven_sil``::

    python test_sil/test_data_driven_sil.py --executable build/data_driven_sil

No pybind11, compiler invocation, or network access is needed by this runner.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_libraries" / "MCAP_python_adaptive_control"))

from data_driven import (  # noqa: E402
    LinearControllerBasis, RationalFilter, SolverOptions, Status,
    frit_cost_gradient, frit_least_squares, frit_optimize, frit_regression,
    vrft_least_squares, vrft_optimize, vrft_regression,
)


def experiment(noise):
    state = 0x12345678
    u = np.zeros(256)
    y = np.zeros(256)
    previous_y = previous_u = 0.0
    for k in range(u.size):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        u[k] = (state >> 16) / 32768.0 - 1.0
        plant_y = 0.7 * previous_y + 0.3 * previous_u
        y[k] = plant_y + noise * (((k * 17 + 3) % 23) - 11) / 11.0
        previous_y, previous_u = plant_y, u[k]
    return u, y


def independent_filter(signal, numerator, denominator):
    """Scalar direct-form I recurrence, independent of library realization."""
    out = np.zeros(len(signal))
    for k in range(len(signal)):
        value = sum(coefficient * signal[k - j]
                    for j, coefficient in enumerate(numerator) if k >= j)
        value -= sum(coefficient * out[k - j]
                     for j, coefficient in enumerate(denominator) if j and k >= j)
        out[k] = value / denominator[0]
    return out


def closed_loop(parameters):
    output = np.zeros(64)
    integral = previous_y = previous_u = 0.0
    for k in range(output.size):
        output[k] = 0.7 * previous_y + 0.3 * previous_u
        error = 1.0 - output[k]
        integral += 0.1 * error
        previous_u = parameters[0] * error + parameters[1] * integral
        previous_y = output[k]
    return output


def close(actual, expected, label, atol=3e-11, rtol=3e-10):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError(f"{label}: nonfinite result")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               equal_nan=False, err_msg=label)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def run_sil(executable):
    completed = subprocess.run([str(Path(executable).resolve())], check=True,
                               capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    if completed.stderr:
        raise AssertionError(f"unexpected C++ diagnostics: {completed.stderr}")
    cases = {
        "exact": ([0.0, 0.25], [1.0, -0.75], 0.0, 0.0, 0),
        "noisy_ridge_discard": ([0.0, 0.25], [1.0, -0.75], 0.003, 0.0003, 11),
        "two_sample_delay": ([0.0, 0.0, 0.25], [1.0, -0.75, 0.0], 0.001, 0.0001, 9),
    }
    if {case["name"] for case in payload["cases"]} != set(cases):
        raise AssertionError("C++ output is missing SIL cases")
    forward = LinearControllerBasis([[1, -1], [0.1, 0]], [1, -1])
    inverse = LinearControllerBasis([[1, 0], [0, 1]], [1, -0.7])
    initial, lower, upper = [0.9, 1.4], [0.1, 0.1], [2, 5]
    max_direct_error = max_iterative_error = 0.0
    for case in payload["cases"]:
        name = case["name"]
        b, a, noise, ridge, discard = cases[name]
        model = RationalFilter(b, a)
        u, y = experiment(noise)
        options = SolverOptions(ridge=ridge, tolerance=1e-9,
                                max_iterations=1000, discard_samples=discard)
        fits = {
            "vrft_least_squares": vrft_least_squares(u, y, model, forward, options),
            "vrft_optimize": vrft_optimize(u, y, model, forward, initial, lower, upper, options),
            "frit_least_squares": frit_least_squares(u, y, model, inverse, options),
            "frit_optimize": frit_optimize(u, y, model, forward, initial, lower, upper, options),
        }
        if set(case["methods"]) != set(fits):
            raise AssertionError(f"{name}: missing algorithms")
        for method, fit in fits.items():
            other = case["methods"][method]
            if fit.status != Status.SUCCESS or other["status"] != "success":
                raise AssertionError(f"{name}/{method}: solver did not converge")
            iterative = method.endswith("optimize")
            difference = close(other["parameters"], fit.parameters, f"{name}/{method}/parameters",
                               atol=2e-4 if iterative else 2e-10,
                               rtol=2e-5 if iterative else 2e-10)
            close(other["cost"], fit.cost, f"{name}/{method}/cost", atol=2e-9, rtol=2e-7)
            if iterative:
                max_iterative_error = max(max_iterative_error, difference)
                if other["iterations"] > options.max_iterations or fit.iterations > options.max_iterations:
                    raise AssertionError("iteration budget exceeded")
            else:
                max_direct_error = max(max_direct_error, difference)
                if other["rank"] != fit.rank or fit.rank != 2:
                    raise AssertionError(f"{name}/{method}: wrong identified rank")

        vx, vt = vrft_regression(u, y, model, forward, options)
        fx, ft = frit_regression(u, y, model, inverse, options)
        close(case["vrft_columns"], vx.T, f"{name}/VRFT regressors")
        close(case["vrft_target"], vt, f"{name}/VRFT target")
        close(case["frit_columns"], fx.T, f"{name}/FRIT regressors")
        close(case["frit_target"], ft, f"{name}/FRIT target")

        # Independent equations also catch a shared implementation mistake.
        filtered_u = independent_filter(u, b, a)
        error = y - independent_filter(y, b, a)
        expected_vx = np.column_stack([error, 0.1 * np.cumsum(error)])
        close(vx, expected_vx[discard:], f"{name}/independent VRFT regressors")
        close(vt, filtered_u[discard:], f"{name}/independent VRFT target")
        inverse_columns = np.column_stack([
            independent_filter(u, [1, 0], [1, -0.7]),
            independent_filter(u, [0, 1], [1, -0.7]),
        ])
        expected_fx = np.column_stack([independent_filter(column, b, a)
                                       for column in inverse_columns.T])
        close(fx, expected_fx[discard:], f"{name}/independent FRIT regressors")
        close(ft, error[discard:], f"{name}/independent FRIT target")
        cost, gradient = frit_cost_gradient(initial, u, y, model, forward, options)
        close(case["frit_initial_cost"], cost, f"{name}/FRIT objective")
        close(case["frit_initial_gradient"], gradient, f"{name}/FRIT gradient")
        for method in ("vrft", "frit"):
            parameters = fits[f"{method}_optimize"].parameters
            close(case[f"{method}_closed_loop"], closed_loop(parameters),
                  f"{name}/{method}/closed-loop trajectory", atol=6e-5, rtol=2e-5)

        if name == "exact":
            for method in ("vrft_least_squares", "vrft_optimize", "frit_optimize"):
                close(fits[method].parameters, [7 / 12, 2.5], f"analytical {method}", atol=2e-4)
            close(fits["frit_least_squares"].parameters, [1.2, -1.2], "analytical inverse FRIT")
            step = 1.0 - 0.75 ** np.arange(64)
            close(closed_loop(fits["vrft_least_squares"].parameters), step, "held-out analytical step")
        print(f"PASS {name}: four algorithms, regressors, objectives, gradient, trajectories")
    print(f"SIL passed: max parameter difference direct={max_direct_error:.3g}, "
          f"iterative={max_iterative_error:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "build" / "data_driven_sil")
    run_sil(parser.parse_args().executable)
