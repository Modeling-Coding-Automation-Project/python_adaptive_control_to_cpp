"""Generate, build, and compare the Python/C++ sample design workflows.

Run with ``python test_sil/test_codegen.py``. Headers and executables are built
in a temporary directory; repository-root generated files are left untouched.
"""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SAMPLE = (ROOT / "external_libraries" / "MCAP_python_adaptive_control" /
                 "sample" / "data_driven")
EXPERIMENT_HEADER = "experiment_time,experiment_input,experiment_output"
SIMULATION_HEADER = (
    "time,reference,desired,y_initial,u_initial,y_direct,u_direct,"
    "y_iterative,u_iterative")


def command(arguments, cwd, expect_success=True):
    result = subprocess.run([str(arg) for arg in arguments], cwd=cwd,
                            capture_output=True, text=True)
    if expect_success and result.returncode:
        raise RuntimeError(f"Command failed: {arguments}\n{result.stdout}\n{result.stderr}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {arguments}")
    return result


def load_demo(method):
    path = ROOT / "sample" / f"{method}_demo.py"
    spec = importlib.util.spec_from_file_location(f"sample_{method}", path)
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
        np.column_stack((design["sample_time"] * np.arange(design["sample_count"]),
                         expected["u"], expected["y"])),
        rtol=1e-12, atol=1e-12, err_msg=f"{method}: experimental input/output")

    direct_name = ("VRFT least squares" if method == "vrft"
                   else "FRIT inverse least squares")
    iterative_name = method.upper() + " PANOC"
    for name in (direct_name, iterative_name):
        match = re.search(
            re.escape(name) + r": parameters=\[([^\]]+)\] status=(\w+) "
            r"cost=(\S+) iterations=(\d+)", stdout)
        if match is None or match.group(2) != "success":
            raise AssertionError(f"{method}: missing successful fit for {name}")
        parameters = np.asarray([float(value) for value in match.group(1).split(",")])
        cost = float(match.group(3))
        if not np.all(np.isfinite(parameters)) or not np.isfinite(cost) or cost < 0:
            raise AssertionError(f"{method}: invalid fit values for {name}")
        np.testing.assert_allclose(parameters, expected["fits"][name].parameters,
                                   rtol=1e-6, atol=1e-5, err_msg=name)
        np.testing.assert_allclose(cost, expected["fits"][name].cost,
                                   rtol=1e-5, atol=1e-9, err_msg=name + " cost")

    actual = read_table(stdout, SIMULATION_HEADER, design["simulation_count"])
    columns = [expected["time"], expected["reference"], expected["desired"]]
    for name in ("initial PI", direct_name, iterative_name):
        columns.extend((expected["trajectories"][name], expected["controls"][name]))
    np.testing.assert_allclose(actual, np.column_stack(columns), rtol=1e-6, atol=1e-7,
                               err_msg=f"{method}: closed-loop outputs and inputs")
    # Independent timing check: y(0)=0, u_direct(0)=1.5, y_direct(1)=0.3.
    # Passing previous_u into an already delayed plant would violate this.
    np.testing.assert_allclose(actual[0, [3, 5, 7]], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual[0, 6], 1.5, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual[1, 5], 0.3, rtol=0, atol=1e-10)


def build_and_compare(workspace, expected):
    (workspace / "sample").mkdir(exist_ok=True)
    for method in ("vrft", "frit"):
        shutil.copyfile(ROOT / "sample" / f"{method}_demo.cpp",
                        workspace / "sample" / f"{method}_demo.cpp")
    # Root-level generated headers specialize the samples' fixed-size types.
    escaped_root = ROOT.as_posix().replace('"', '\\"')
    (workspace / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.16)\n'
        'project(generated_control_validation LANGUAGES CXX)\n'
        'set(ADAPTIVE_CONTROL_BUILD_TESTS OFF CACHE BOOL "" FORCE)\n'
        'set(ADAPTIVE_CONTROL_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)\n'
        f'add_subdirectory("{escaped_root}" repository)\n'
        'foreach(method vrft frit)\n'
        '  add_executable(${method} sample/${method}_demo.cpp)\n'
        '  target_link_libraries(${method} PRIVATE MCAP::AdaptiveControl)\n'
        '  target_include_directories(${method} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})\n'
        '  set_target_properties(${method} PROPERTIES CXX_STANDARD 11 '
        'CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)\n'
        '  target_compile_options(${method} PRIVATE -Wall -Wextra -Wpedantic -Werror)\n'
        'endforeach()\n')
    build = workspace / "build"
    command(["cmake", "-S", workspace, "-B", build,
             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=g++"], workspace)
    command(["cmake", "--build", build, "--parallel", "2"], workspace)
    for method in ("vrft", "frit"):
        stdout = command([build / method], workspace).stdout
        compare_workflow(method, stdout, expected[method])


def run_codegen_checks():
    with tempfile.TemporaryDirectory(prefix="data-driven-codegen-") as directory:
        workspace = Path(directory)
        demos, expected = {}, {}
        for method in ("vrft", "frit"):
            source = ROOT / "sample" / f"{method}_demo.py"
            counterpart = PYTHON_SAMPLE / source.name
            if source.read_bytes() != counterpart.read_bytes():
                raise AssertionError(f"The two {source.name} copies differ")
            arguments = [sys.executable, source, "--no-plot", "--output-dir", workspace]
            command(arguments, workspace)
            header = workspace / f"{method}_demo.hpp"
            original = header.read_bytes()  # Missing generation is a failure.
            command(arguments + ["--check-generated"], workspace)
            header.write_bytes(original + b"\n// simulate a stale generated artifact\n")
            command(arguments + ["--check-generated"], workspace, expect_success=False)
            command(arguments, workspace)
            if header.read_bytes() != original:
                raise AssertionError(f"{method} generation is nondeterministic")
            demos[method] = load_demo(method)
            expected[method] = demos[method].run_demo()
        build_and_compare(workspace, expected)

        # Change compile-time sizes and filter capacity without changing the
        # plant dynamics. Generation must work before any data collection/fit.
        variant = workspace / "different_structure"
        variant.mkdir()
        for method, demo in demos.items():
            design = demo.create_design()
            design["sample_count"] = 96
            design["simulation_count"] = 60
            model = design["model"]
            design["model"] = type(model)(np.pad(model.numerator, (0, 1)),
                                           np.pad(model.denominator, (0, 1)))
            demo.generate_cpp(design, variant)
            header = (variant / f"{method}_demo.hpp").read_text()
            for declaration in ("SAMPLE_COUNT = 96;", "SIMULATION_COUNT = 60;", "ORDER = 2;"):
                if declaration not in header:
                    raise AssertionError(f"{method}: missing generated {declaration}")
            expected[method] = demo.run_demo(design)
        build_and_compare(variant, expected)
    print("Code generation passed: identical samples, isolated execution, stale-header "
          "detection, deterministic configuration generation, and Python/C++ "
          "experiment/tuning/closed-loop agreement for both problem sizes")


if __name__ == "__main__":
    run_codegen_checks()
