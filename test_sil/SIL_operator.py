"""Build and load one sample's C++ SIL module, as in python_control_to_cpp.

Each method directory owns its Python workflow and <method>_SIL.cpp wrapper.
Only the pybind11 test module is built here; no controller code is generated.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import pybind11


ROOT = Path(__file__).resolve().parents[1]


class CmakeGenerator:
    def __init__(self, sil_folder):
        self.sil_folder = Path(sil_folder).resolve()
        self.module_name = self.sil_folder.name.title() + "SIL"

    def generate_cmake_lists_txt(self, destination):
        # Use absolute paths so invoking a SIL script from another cwd works.
        root = ROOT.as_posix().replace('"', '\\"')
        source = (self.sil_folder / (self.sil_folder.name + "_SIL.cpp"))
        source = source.as_posix().replace('"', '\\"')
        text = (
            "cmake_minimum_required(VERSION 3.18)\n"
            f"project({self.module_name} LANGUAGES CXX)\n"
            "set(PYBIND11_FINDPYTHON ON)\n"
            "find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)\n"
            "find_package(pybind11 CONFIG REQUIRED)\n"
            "set(ADAPTIVE_CONTROL_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)\n"
            "set(ADAPTIVE_CONTROL_BUILD_SAMPLES OFF CACHE BOOL \"\" FORCE)\n"
            f"add_subdirectory(\"{root}\" repository)\n"
            f"pybind11_add_module({self.module_name} NO_EXTRAS \"{source}\")\n"
            f"target_link_libraries({self.module_name} PRIVATE MCAP::AdaptiveControl)\n"
            f"target_include_directories({self.module_name} PRIVATE \"{root}/test_sil\")\n"
            f"set_target_properties({self.module_name} PROPERTIES\n"
            "  CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO\n"
            "  LIBRARY_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}\")\n"
            f"target_compile_options({self.module_name} PRIVATE -O2 -Wall -Wextra -Werror)\n"
        )
        (Path(destination) / "CMakeLists.txt").write_text(text, encoding="utf-8")


class SIL_Builder:
    """Context manager returning the method's freshly built pybind11 module.

    The active Python interpreter supplies both the ABI and pybind11 package.
    All generated CMake/build files live in a temporary directory.
    """

    def __init__(self, sil_folder):
        self.generator = CmakeGenerator(sil_folder)
        self.temporary = None

    @staticmethod
    def _run(arguments):
        completed = subprocess.run(arguments, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(
                f"SIL build failed: {arguments}\n{completed.stdout}")

    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=self.generator.module_name + "-")
        directory = Path(self.temporary.name)
        try:
            self.generator.generate_cmake_lists_txt(directory)
            build = directory / "build"
            self._run(["cmake", "-S", str(directory), "-B", str(build),
                       "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=g++",
                       f"-DPython_EXECUTABLE={sys.executable}",
                       f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"])
            self._run(["cmake", "--build", str(build), "--parallel", "2"])
            module_name = self.generator.module_name
            library = build / \
                (module_name + sysconfig.get_config_var("EXT_SUFFIX"))
            spec = importlib.util.spec_from_file_location(module_name, library)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except BaseException:
            self.temporary.cleanup()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.temporary.cleanup()
