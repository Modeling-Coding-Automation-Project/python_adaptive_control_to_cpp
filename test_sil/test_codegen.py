"""Verify paired Python samples and compile their freshly generated C++11 code.

Run with ``python test_sil/test_codegen.py``. Temporary generation and builds
leave the checked-in headers untouched and exercise execution from another cwd.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SAMPLE = (ROOT / "external_libraries" / "MCAP_python_adaptive_control" /
                 "sample" / "data_driven")


def command(arguments, cwd, expect_success=True):
    result = subprocess.run([str(arg) for arg in arguments], cwd=cwd,
                            capture_output=True, text=True)
    if expect_success and result.returncode:
        raise RuntimeError(f"Command failed: {arguments}\n{result.stdout}\n{result.stderr}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {arguments}")
    return result


def run_codegen_checks():
    with tempfile.TemporaryDirectory(prefix="data-driven-codegen-") as directory:
        workspace = Path(directory)
        # Mirror the repository layout: generated headers live beside sample/.
        generated = workspace
        (workspace / "sample").mkdir()
        for method in ("vrft", "frit"):
            source = ROOT / "sample" / f"{method}_demo.py"
            counterpart = PYTHON_SAMPLE / source.name
            if source.read_bytes() != counterpart.read_bytes():
                raise AssertionError(f"The two {source.name} copies differ")
            arguments = [sys.executable, source, "--no-plot", "--output-dir", generated]
            command(arguments, workspace)
            header = generated / f"{method}_demo.hpp"
            original = header.read_bytes()  # Missing generation is a failure.
            command(arguments + ["--check-generated"], workspace)
            header.write_bytes(original + b"\n// simulate a stale generated artifact\n")
            command(arguments + ["--check-generated"], workspace, expect_success=False)
            command(arguments, workspace)
            if header.read_bytes() != original:
                raise AssertionError(f"{method} generation is nondeterministic")
            shutil.copyfile(ROOT / "sample" / f"{method}_demo.cpp",
                            workspace / "sample" / f"{method}_demo.cpp")

        shutil.copyfile(ROOT / "sample" / "data_driven_sample_common.hpp",
                        workspace / "sample" / "data_driven_sample_common.hpp")
        # Reuse the library's declared include dependencies; compile the copied
        # demos so their ../ includes resolve to the freshly generated headers.
        escaped_root = ROOT.as_posix().replace('"', '\\"')
        (workspace / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.16)\n'
            'project(generated_control_validation LANGUAGES CXX)\n'
            'set(ADAPTIVE_CONTROL_BUILD_TESTS OFF CACHE BOOL "" FORCE)\n'
            'set(ADAPTIVE_CONTROL_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)\n'
            f'add_subdirectory("{escaped_root}" repository)\n'
            'enable_testing()\n'
            'foreach(method vrft frit)\n'
            '  add_executable(${method} sample/${method}_demo.cpp)\n'
            '  target_link_libraries(${method} PRIVATE MCAP::AdaptiveControl)\n'
            '  set_target_properties(${method} PROPERTIES CXX_STANDARD 11 '
            'CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)\n'
            '  add_test(NAME ${method}_generated_demo COMMAND ${method})\n'
            '  string(TOUPPER "${method}" method_name)\n'
            '  set_tests_properties(${method}_generated_demo PROPERTIES\n'
            '    PASS_REGULAR_EXPRESSION "${method_name} result: PASS"\n'
            '    FAIL_REGULAR_EXPRESSION "FAIL")\n'
            'endforeach()\n')
        build = workspace / "build"
        command(["cmake", "-S", workspace, "-B", build,
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=g++"], workspace)
        command(["cmake", "--build", build, "--parallel", "2"], workspace)
        command(["ctest", "--test-dir", build, "--output-on-failure"], workspace)
    print("Code generation passed: identical samples, isolated execution, stale-header "
          "detection, deterministic generation, and both generated C++11 demos")


if __name__ == "__main__":
    run_codegen_checks()
