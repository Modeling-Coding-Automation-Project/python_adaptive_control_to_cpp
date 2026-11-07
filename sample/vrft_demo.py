"""VRFT design workflow: collect data, tune controllers, and simulate feedback.

Python generates the C++ problem configuration before tuning, then runs the
same experiment and feedback simulation as vrft_demo.cpp. Python plots the
time series with Dash; the C++ example prints them to the terminal.
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


def create_design():
    """1. Choose the experiment, target response, and controller structure."""
    sample_time = .1
    model = adaptive_control.RationalFilter([0., .3], [1., -.7])
    basis = adaptive_control.LinearControllerBasis.pi(sample_time)

    return {
        "sample_time": sample_time, "sample_count": 128,
        "simulation_count": 100, "excitation_seed": 0x12345678,
        # Simulated plant: x(k+1)=.8*x(k)+.2*u(k), y(k)=x(k).
        # This plant is only used to collect data and evaluate the controllers.
        "plant_a": .8, "plant_b": .2,
        "model": model, "basis": basis, "inverse_basis": None,
        "initial": np.array([.7, 1.0]),
        "lower": np.array([.1, .1]), "upper": np.array([3., 6.]),
        "options": adaptive_control.SolverOptions(
            tolerance=2e-5, max_iterations=1000),
    }


def generate_cpp(design, output_dir=None, check=False):
    """Generate fixed-size C++ types and settings before running the experiment."""
    deploy_module = REPOSITORY / "data_driven" / "data_driven_deploy.py"
    if not deploy_module.is_file():
        if output_dir is not None or check:
            raise RuntimeError(
                "C++ generation is available in python_adaptive_control_to_cpp")
        return None
    from data_driven.data_driven_deploy import DataDrivenDeploy
    destination = REPOSITORY if output_dir is None else Path(output_dir)
    path = DataDrivenDeploy.generate_vrft_cpp_code(
        design, destination, check=check)
    print(f"C++ header {'verified' if check else 'generated'}: {path}")
    return path


def run_demo(design=None):
    design = create_design() if design is None else design
    model, basis, options = design["model"], design["basis"], design["options"]

    # 2. Collect one input/output experiment from a zero-state plant.
    # Replace this loop with measured u,y when tuning a physical system.
    u = np.zeros(design["sample_count"])
    y = np.zeros(design["sample_count"])
    random_state = design["excitation_seed"]
    plant_state = 0.0
    for index in range(u.size):
        random_state = (1664525 * random_state + 1013904223) & 0xFFFFFFFF
        u[index] = (random_state >> 16) / 32768.0 - 1.0
        y[index] = plant_state
        plant_state = design["plant_a"] * \
            plant_state + design["plant_b"] * u[index]

    # 3. Tune from the recorded data, target model, and controller basis.
    # Neither tuning method receives the simulated plant coefficients.
    tuner = adaptive_control.VRFT(model, basis)
    direct = tuner.fit_least_squares(u, y, options)
    iterative = tuner.fit_iterative(
        u, y, design["initial"], design["lower"], design["upper"], options)
    fits = {"VRFT least squares": direct, "VRFT PANOC": iterative}
    if not all(fit.success for fit in fits.values()):
        raise RuntimeError("Tuning failed: " + ", ".join(
            f"{name}={fit.status.value}" for name, fit in fits.items()))

    # 4. Build the initial and newly tuned feedback controllers.
    controllers = {
        "initial PI": basis.controller(design["initial"]),
        "VRFT least squares": basis.controller(direct.parameters),
        "VRFT PANOC": basis.controller(iterative.parameters),
    }

    # 5. Run independent closed loops from zero state with a unit-step command.
    # At time k: measure y(k), compute u(k), then advance the plant to x(k+1).
    reference = np.ones(design["simulation_count"])
    desired = np.zeros(reference.size)
    trajectories = {name: np.zeros(reference.size) for name in controllers}
    controls = {name: np.zeros(reference.size) for name in controllers}
    controller_states = {name: controller.state()
                         for name, controller in controllers.items()}
    plant_states = {name: 0.0 for name in controllers}
    model_state = model.state()
    for index, command in enumerate(reference):
        desired[index] = model_state.step(command)
        for name, controller_state in controller_states.items():
            measured_output = plant_states[name]
            control = controller_state.step(command - measured_output)
            trajectories[name][index] = measured_output
            controls[name][index] = control
            plant_states[name] = (design["plant_a"] * measured_output
                                  + design["plant_b"] * control)

    return {"design": design, "fits": fits, "u": u, "y": y,
            "time": design["sample_time"] * np.arange(reference.size),
            "reference": reference, "desired": desired,
            "trajectories": trajectories, "controls": controls}


def plot_demo(result):
    """6. Display the experiment, tracking response, and control input."""
    from sample.simulation_manager.visualize.simulation_plotter_dash import SimulationPlotterDash
    plotter = SimulationPlotterDash(activate_dump=False)
    time = result["time"].reshape(-1, 1)
    outputs = {"reference": result["reference"], "desired": result["desired"],
               **result["trajectories"]}
    for name, signal in outputs.items():
        plotter.append_sequence_name(signal.reshape(-1, 1), name)
        plotter.assign(name, position=(0, 0), x_sequence=time,
                       x_sequence_name="time [s]", label=name)
    for name, signal in result["controls"].items():
        signal_name = "control: " + name
        plotter.append_sequence_name(signal.reshape(-1, 1), signal_name)
        plotter.assign(signal_name, position=(1, 0), x_sequence=time,
                       x_sequence_name="time [s]", label=signal_name)
    experiment_time = (result["design"]["sample_time"]
                       * np.arange(result["u"].size)).reshape(-1, 1)
    for row, (name, signal) in enumerate((
            ("recorded input", result["u"]), ("recorded output", result["y"]))):
        plotter.append_sequence_name(signal.reshape(-1, 1), name)
        plotter.assign(name, position=(row, 1), x_sequence=experiment_time,
                       x_sequence_name="experiment time [s]", label=name)
    plotter.plot("VRFT controller design")
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

    design = create_design()
    generate_cpp(design, args.output_dir, args.check_generated)
    result = run_demo(design)
    print(f"VRFT experiment: {design['sample_count']} input/output samples, "
          f"sample time={design['sample_time']} s")
    for name, fit in result["fits"].items():
        print(f"{name}: parameters={fit.parameters}, status={fit.status.value}, "
              f"cost={fit.cost:.10g}, iterations={fit.iterations}")
    if not args.no_plot:
        plot_demo(result)


if __name__ == "__main__":
    main()
