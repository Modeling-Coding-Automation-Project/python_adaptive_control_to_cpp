"""Compare standalone C++ demos with the Python reference design workflows.

Run ``python test_sil/test_samples.py`` to build the C++ samples first, without
running Python design code or generating headers. Use ``--build-dir build`` to
compare already-built ``vrft_demo`` and ``frit_demo`` executables instead.
"""
import argparse
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "external_libraries" / "MCAP_python_adaptive_control"
EXPERIMENT_HEADER = (
    "experiment_time,experiment_reference,experiment_input,experiment_output")
SIMULATION_HEADER = (
    "time,reference,desired,y_initial,u_initial,y_direct,u_direct,"
    "y_iterative,u_iterative")


def command(arguments, cwd):
    result = subprocess.run([str(arg) for arg in arguments], cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {arguments}\n{result.stdout}\n{result.stderr}")
    return result


def load_reference(method):
    # The samples import their own data_driven package and simulation_manager.
    # The C++ repository's data_driven directory contains no Python package.
    sys.path.insert(0, str(PYTHON_ROOT))
    path = PYTHON_ROOT / "sample" / "data_driven" / f"{method}_demo.py"
    spec = importlib.util.spec_from_file_location(f"reference_{method}_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_table(stdout, header, sample_count):
    lines = stdout.splitlines()
    if lines.count(header) != 1:
        raise AssertionError(f"Expected one table headed {header!r}")
    columns = len(header.split(","))
    rows = []
    for line in lines[lines.index(header) + 1:]:
        fields = line.split(",")
        if len(fields) != columns:
            break
        try:
            rows.append([float(value) for value in fields])
        except ValueError:
            break
    values = np.asarray(rows)
    if values.shape != (sample_count, columns) or not np.all(np.isfinite(values)):
        raise AssertionError(f"Incomplete or nonfinite {header!r} table: {values.shape}")
    return values


def compare_workflow(method, stdout, expected):
    if "ERROR:" in stdout or f"{method.upper()} simulation complete." not in stdout:
        raise AssertionError(f"The {method} workflow did not complete:\n{stdout}")
    design = expected["design"]
    experiment = read_table(stdout, EXPERIMENT_HEADER, design["sample_count"])
    np.testing.assert_allclose(
        experiment,
        np.column_stack((expected["experiment_time"], expected["experiment_reference"],
                         expected["u"], expected["y"])),
        rtol=1e-12, atol=1e-12, equal_nan=False,
        err_msg=f"{method}: closed-loop experiment with the fixed initial PI")
    # Independently distinguish the initial-PI experiment from open-loop noise.
    np.testing.assert_allclose(experiment[:, 1], 1.0, rtol=0, atol=0)
    np.testing.assert_allclose(experiment[0, 2:], [0.8, 0.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(experiment[1, 3], 0.16, rtol=0, atol=1e-12)

    direct_name = ("VRFT least squares" if method == "vrft"
                   else "FRIT inverse least squares")
    iterative_name = method.upper() + " PANOC"
    for name in (direct_name, iterative_name):
        fit = expected["fits"][name]
        match = re.search(
            re.escape(name) + r": parameters=\[([^\]]+)\] status=(\w+) "
            r"cost=(\S+) iterations=(\d+)", stdout)
        if not fit.success or match is None or match.group(2) != "success":
            raise AssertionError(f"{method}: missing successful fit for {name}")
        parameters = np.asarray([float(value) for value in match.group(1).split(",")])
        cost = float(match.group(3))
        iterations = int(match.group(4))
        if not np.all(np.isfinite(parameters)) or not np.isfinite(cost) or cost < 0:
            raise AssertionError(f"{method}: invalid fit values for {name}")
        if iterations > design["options"].max_iterations:
            raise AssertionError(f"{method}: iteration budget exceeded for {name}")
        np.testing.assert_allclose(parameters, fit.parameters,
                                   rtol=1e-6, atol=1e-5, equal_nan=False, err_msg=name)
        np.testing.assert_allclose(cost, fit.cost, rtol=1e-5, atol=1e-9,
                                   equal_nan=False, err_msg=name + " cost")

    actual = read_table(stdout, SIMULATION_HEADER, design["simulation_count"])
    columns = [expected["time"], expected["reference"], expected["desired"]]
    controller_names = ("initial PI", direct_name, iterative_name)
    for name in controller_names:
        columns.extend((expected["trajectories"][name], expected["controls"][name]))
    np.testing.assert_allclose(actual, np.column_stack(columns), rtol=1e-6, atol=1e-7,
                               equal_nan=False,
                               err_msg=f"{method}: closed-loop outputs and inputs")
    # Fresh zero states and one sample of delay: y(0)=0, u_direct(0)=1.5,
    # y_direct(1)=0.3. Reusing the acquisition state would violate these values.
    np.testing.assert_allclose(actual[0, [3, 5, 7]], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual[0, 6], 1.5, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual[1, 5], 0.3, rtol=0, atol=1e-10)

    for name, output_column in zip(controller_names, (3, 5, 7)):
        match = re.search(re.escape(name) + r": RMSE=(\S+)", stdout)
        if match is None:
            raise AssertionError(f"{method}: missing tracking RMSE for {name}")
        rmse = float(match.group(1))
        if not np.isfinite(rmse) or rmse < 0:
            raise AssertionError(f"{method}: invalid tracking RMSE for {name}")
        expected_rmse = np.sqrt(np.mean(
            (expected["trajectories"][name] - expected["desired"]) ** 2))
        np.testing.assert_allclose(rmse, expected_rmse, rtol=1e-4, atol=1e-7,
                                   equal_nan=False, err_msg=name + " RMSE")
        # PythonMath uses an approximate Newton square root (the observed
        # relative error is about 6.4e-8 for this FRIT record). Allow 1e-6 here,
        # while still detecting the 1e-5 floor of an unscaled square root.
        series_rmse = np.sqrt(np.mean((actual[:, output_column] - actual[:, 2]) ** 2))
        np.testing.assert_allclose(rmse, series_rmse, rtol=1e-6, atol=1e-13,
                                   equal_nan=False, err_msg=name + " reported RMSE")


def build_samples(build_dir, cwd):
    command(["cmake", "-S", ROOT, "-B", build_dir,
             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=g++",
             "-DADAPTIVE_CONTROL_BUILD_TESTS=OFF",
             "-DADAPTIVE_CONTROL_BUILD_SAMPLES=ON"], cwd)
    command(["cmake", "--build", build_dir, "--parallel", "2"], cwd)


def run_sample_checks(build_dir=None):
    with tempfile.TemporaryDirectory(prefix="data-driven-samples-") as directory:
        workspace = Path(directory)
        if build_dir is None:
            build_dir = workspace / "build"
            # No Python design or code generation is run before this C++ build.
            build_samples(build_dir, workspace)
        else:
            build_dir = Path(build_dir).resolve()
        for method in ("vrft", "frit"):
            completed = command([build_dir / f"{method}_demo"], workspace)
            if completed.stderr:
                raise AssertionError(f"{method}: unexpected diagnostics: {completed.stderr}")
            expected = load_reference(method).run_demo()
            compare_workflow(method, completed.stdout, expected)
            print(f"PASS {method}: initial-PI experiment, offline tuning, "
                  "fresh closed-loop responses and tracking RMSE")
    print("Sample SIL passed: standalone C++ execution agrees with the Python reference demos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path,
                        help="directory containing already-built vrft_demo and frit_demo executables")
    run_sample_checks(parser.parse_args().build_dir)
