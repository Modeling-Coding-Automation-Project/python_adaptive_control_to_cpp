"""Shared comparisons and recorded-data cases; each method owns its workflow."""
import numpy as np


def expect_near(tester, actual, expected, tolerance, label):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label}: shape {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError(f"{label}: nonfinite result")
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    # Scalar comparison keeps the supplied tolerance absolute, including arrays.
    tester.expect_near(error, 0.0, tolerance,
                       f"{label}: max absolute error {error:.3g}")
    return error


def make_cpp_options(cpp, options):
    result = cpp.SolverOptions()
    for name in ("ridge", "tolerance", "rank_tolerance",
                 "max_iterations", "discard_samples"):
        setattr(result, name, getattr(options, name))
    return result


def compare_fit(tester, actual, expected, options, label,
                parameter_tolerance=1e-5, cost_tolerance=2e-9):
    if not expected.success or not actual.success or actual.status != expected.status.value:
        raise AssertionError(
            f"{label}: Python={expected.status.value}, C++={actual.status}")
    if actual.rank != expected.rank:
        raise AssertionError(
            f"{label}: Python rank={expected.rank}, C++ rank={actual.rank}")
    if not (0 <= actual.iterations <= options.max_iterations
            and 0 <= expected.iterations <= options.max_iterations):
        raise AssertionError(f"{label}: iteration budget exceeded")
    if actual.cost < 0 or expected.cost < 0:
        raise AssertionError(f"{label}: negative squared-error cost")
    expect_near(tester, actual.parameters, expected.parameters,
                parameter_tolerance, label + " parameters")
    expect_near(tester, actual.cost, expected.cost,
                cost_tolerance, label + " cost")


def regression_cases():
    yield "exact", [0.0, 0.25], [1.0, -0.75], 0.0, 0.0, 0
    yield "noisy_ridge_discard", [0.0, 0.25], [1.0, -0.75], 0.003, 0.0003, 11
    yield "two_sample_delay", [0.0, 0.0, 0.25], [1.0, -0.75, 0.0], 0.001, 0.0001, 9


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
    """Direct-form I equation, independent of the library's realization."""
    output = np.zeros(len(signal))
    for k in range(len(signal)):
        value = sum(coefficient * signal[k - j]
                    for j, coefficient in enumerate(numerator) if k >= j)
        value -= sum(coefficient * output[k - j]
                     for j, coefficient in enumerate(denominator) if j and k >= j)
        output[k] = value / denominator[0]
    return output
