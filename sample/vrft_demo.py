"""VRFT tuning, interactive response plots, and C++ configuration generation.

Run normally to generate C++ (in the C++ repository) and open the Dash plot.
Use --no-plot for automated runs. This file is identical in both repositories.
"""
import argparse
from importlib import import_module
from pathlib import Path
import sys

import numpy as np


# The C++ checkout uses sample/, and the Python checkout uses sample/data_driven/.
SAMPLE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = (SAMPLE_DIRECTORY.parent if SAMPLE_DIRECTORY.name == "sample"
              else SAMPLE_DIRECTORY.parents[1])
sys.path.insert(0, str(REPOSITORY))
if (REPOSITORY / "CMakeLists.txt").is_file():
    from external_libraries.MCAP_python_adaptive_control import data_driven as adaptive_control
else:
    # In the standalone Python checkout, data_driven is the library itself.
    adaptive_control = import_module("data_driven")


def closed_loop(controller, reference):
    """Validation plant only: G=.2 z^-1/(1-.8 z^-1)."""
    state = controller.state()
    response = np.zeros(len(reference))
    output = 0.0
    for index, command in enumerate(reference):
        response[index] = output
        control = state.step(command - output)
        output = .8 * output + .2 * control
    return response


def run_demo():
    sample_time = .1
    # Record one reproducible input/output experiment. The tuning functions
    # receive only u,y; they receive no plant model.
    u = np.zeros(128)
    random_state = 0x12345678
    for index in range(u.size):
        random_state = (1664525 * random_state + 1013904223) & 0xFFFFFFFF
        u[index] = (random_state >> 16) / 32768.0 - 1.0
    plant = adaptive_control.RationalFilter([0., .2], [1., -.8])
    y = plant.filter(u)
    model = adaptive_control.RationalFilter([0., .3], [1., -.7])
    basis = adaptive_control.LinearControllerBasis.pi(sample_time)
    initial = np.array([.7, 1.0])
    lower, upper = np.array([.1, .1]), np.array([3., 6.])
    # Use the same practical residual tolerance as the paired C++ deployment.
    options = adaptive_control.SolverOptions(
        tolerance=2e-5, max_iterations=1000)
    fits = {
        "VRFT least squares": adaptive_control.vrft_least_squares(
            u, y, model, basis, options),
        "VRFT PANOC": adaptive_control.vrft_optimize(
            u, y, model, basis, initial, lower, upper, options),
    }
    if not all(result.success for result in fits.values()):
        raise RuntimeError("Tuning failed: " + ", ".join(
            f"{name}={fit.status.value}" for name, fit in fits.items()))

    reference = np.ones(100)
    desired = model.filter(reference)
    trajectories = {"initial PI": closed_loop(
        basis.controller(initial), reference)}
    for name, fit in fits.items():
        controller = basis.controller(fit.parameters)
        trajectories[name] = closed_loop(controller, reference)
    return {"fits": fits, "time": sample_time * np.arange(reference.size),
            "reference": reference, "desired": desired, "trajectories": trajectories,
            "u": u, "y": y, "model": model, "basis": basis,
            "inverse_basis": None, "options": options,
            "initial": initial, "lower": lower, "upper": upper}


def generate_cpp(result, output_dir=None, check=False):
    # Like the MPC demos, deploy Python model/controller objects into C++
    # construction helpers. Never import or execute another checkout's sample.
    deploy_module = REPOSITORY / "data_driven" / "data_driven_deploy.py"
    if not deploy_module.is_file():
        if output_dir is not None or check:
            raise RuntimeError(
                "C++ generation is available in python_adaptive_control_to_cpp")
        return None
    from data_driven.data_driven_deploy import DataDrivenDeploy
    destination = REPOSITORY if output_dir is None else Path(output_dir)
    path = DataDrivenDeploy.generate_vrft_cpp_code(
        result, destination, check=check)
    print(f"C++ header {'verified' if check else 'generated'}: {path}")
    return path


def plot_demo(result):
    # Use this checkout's simulation_manager, including when run from elsewhere.
    from sample.simulation_manager.visualize.simulation_plotter_dash import SimulationPlotterDash
    plotter = SimulationPlotterDash(activate_dump=False)
    signals = {"reference": result["reference"], "desired": result["desired"],
               **result["trajectories"]}
    for name, signal in signals.items():
        plotter.append_sequence_name(signal.reshape(-1, 1), name)
        plotter.assign(name, position=(0, 0), row=0, column=0,
                       x_sequence=result["time"].reshape(-1, 1), label=name)
    plotter.plot("VRFT PI tuning")
    return plotter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true",
                        help="skip interactive HTML plotting for automated runs")
    parser.add_argument("--output-dir", type=Path,
                        help="C++ generated-header destination (default: repository root)")
    parser.add_argument("--check-generated", action="store_true",
                        help="verify the generated header without changing it")
    args = parser.parse_args()
    result = run_demo()
    for name, fit in result["fits"].items():
        rmse = np.sqrt(
            np.mean((result["trajectories"][name] - result["desired"])**2))
        print(f"{name}: rho={fit.parameters}, status={fit.status.value}, "
              f"cost={fit.cost:.3e}, iterations={fit.iterations}, step RMSE={rmse:.3e}")
    generate_cpp(result, args.output_dir, args.check_generated)
    if not args.no_plot:
        plot_demo(result)


if __name__ == "__main__":
    main()
