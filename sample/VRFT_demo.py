"""
VRFT (Virtual Reference Feedback Tuning) demo.

Reference URL:
https://qiita.com/larking95/items/70f88b30072e720f58d9
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import scipy
import control
import matplotlib.pyplot as plt

from python_adaptive_control.vrft import VRFT
from python_adaptive_control.pid_controller import DiscretePID_Controller
from sample.simulation_plotter import SimulationPlotter
from sample.sampler import Sampler

# design requirements
# sample time
Ts = 0.001
Te = 10
number_of_samples = int(Te / Ts) + 1

Te_sim = 2.0
number_of_samples_sim = int(Te_sim / Ts) + 1

# reference model
tau = 0.2
M = control.c2d(control.tf([1], [tau, 1]), Ts)
# M = control.c2d(control.tf([50], [1, 10, 50]), Ts)
T_M, yout_M = control.step_response(M, Te_sim)

# PID controller
Kp_old = 1.0
Ki_old = 1.0
Kd_old = 0.1
pid_controller_old = DiscretePID_Controller(Ts, Kp_old, Ki_old, Kd_old)

# input signal
input_points = np.array([
    [0.0, 1.0],
    [2.0, 1.0],
    [2.0, 0.0],
    [4.0, 0.0],
    [4.0, 1.0],
    [6.0, 1.0],
    [6.0, 0.0],
    [8.0, 0.0],
    [8.0, 1.0],
    [10.0, 1.0]
])

_, input_values = Sampler.create_periodical(input_points, 0.0, Te, Ts)

# get input and output signals
# M sequence signal
n = 15
T = 2**n - 1
p = 15
N = T * p
u0, _ = scipy.signal.max_len_seq(n, length=N)
u0 = -(2.0 * u0 - 1.0)
prbs_input = u0[0:number_of_samples]
prbs_input = prbs_input.reshape((number_of_samples, 1))


# plant model
plant = control.tf([1], [1, 1])
# plant = control.tf([1.96], [1, 1.98, 1.96])
plant_d = control.c2d(plant, Ts)
plant_ss = control.ss(plant_d)

# simulate and get plant output
t0 = []
u0 = []
y0 = []
X0 = np.zeros((plant_ss.A.shape[0], 1))

result_plotter = SimulationPlotter()

y = np.array([[0.0]])
for i in range(number_of_samples):
    t0.append(i * Ts)

    r = input_values[i, 0]

    # controller
    e = r - y
    u = pid_controller_old.update(e)

    # u = np.array([[prbs_input[i]]])

    # plant output
    y = plant_ss.C @ X0 + plant_ss.D @ u
    X0 = plant_ss.A @ X0 + plant_ss.B @ u

    u0.append(u)
    y0.append(y)

    result_plotter.append_name(r, "r")

t0 = np.array(t0).squeeze()
u0 = np.array(u0).squeeze()
y0 = np.array(y0).squeeze()

result_plotter.append_sequence(u0)
result_plotter.append_sequence(y0)
result_plotter.append_sequence_name(yout_M, "yout_M")

result_plotter.assign("r", column=0, row=0, position=(0, 0),
                      x_sequence=t0, label="r", line_style="--")
result_plotter.assign("yout_M", column=0, row=0, position=(0, 0),
                      x_sequence=T_M, label="ref_model", line_style="--")
result_plotter.assign("y0", column=0, row=0, position=(0, 0),
                      x_sequence=t0, label="y0")
result_plotter.assign("u0", column=0, row=0, position=(1, 0),
                      x_sequence=t0, label="u0")

# design VRFT
vrft = VRFT(Ts, M, number_of_samples)

debug_flag = True

if debug_flag:
    ul = vrft.simulate_prefilter(u0)

    result_plotter.append_sequence_name(ul, "ul")
    result_plotter.assign("ul", column=0, row=0, position=(0, 1),
                          x_sequence=t0, label="ul")

    el = vrft.simulate_pseudo_error(y0)

    result_plotter.append_sequence_name(el, "el")
    result_plotter.assign("el", column=0, row=0, position=(1, 1),
                          x_sequence=t0, label="el")

    phi = vrft.simulate_controller_structure(el)

    result_plotter.append_sequence_name(phi, "phi")
    result_plotter.assign("phi", column=0, row=0, position=(0, 2),
                          x_sequence=t0, label="phi_P_term")
    result_plotter.assign("phi", column=1, row=0, position=(1, 2),
                          x_sequence=t0, label="phi_I_term")
    result_plotter.assign("phi", column=2, row=0, position=(2, 2),
                          x_sequence=t0, label="phi_D_term")

    rho = vrft.estimate_gains(phi, ul)

else:
    rho = vrft.solve(u0, y0)

print("rho = ", rho)

# design new pid controller
Kp_new = rho[0, 0]
Ki_new = rho[1, 0]
Kd_new = rho[2, 0]
pid_controller_new = DiscretePID_Controller(Ts, Kp_new, Ki_new, Kd_new)

# simulate and compare results


X0 = np.zeros((plant_ss.A.shape[0], 1))

ref = np.array([[1.0]])
y_old = np.array([[0.0]])
# simulate conventional PID controller
for i in range(number_of_samples_sim):
    # controller input
    e = ref - y_old

    # controller output
    u = pid_controller_old.update(e)

    # plant output
    y_old = plant_ss.C @ X0 + plant_ss.D @ u
    X0 = plant_ss.A @ X0 + plant_ss.B @ u

    result_plotter.append_name(ref, "ref")
    result_plotter.append_name(y_old, "y_old")

# simulate new PID controller
y_new = np.array([[0.0]])
X0 = np.zeros((plant_ss.A.shape[0], 1))
for i in range(number_of_samples_sim):
    # controller input
    e = ref - y_new

    # controller output
    u = pid_controller_new.update(e)

    # plant output
    y_new = plant_ss.C @ X0 + plant_ss.D @ u
    X0 = plant_ss.A @ X0 + plant_ss.B @ u

    result_plotter.append_name(y_new, "y_new")

result_plotter.assign("ref", column=0, row=0, position=(2, 0),
                      x_sequence=t0, label="ref", line_style="--")
result_plotter.assign("yout_M", column=0, row=0, position=(2, 0),
                      x_sequence=t0, label="ref_model", line_style="--")
result_plotter.assign("y_old", column=0, row=0, position=(2, 0),
                      x_sequence=t0, label="y_old")
result_plotter.assign("y_new", column=0, row=0, position=(2, 0),
                      x_sequence=t0, label="y_new")

result_plotter.pre_plot("simulate and compare results")

# plot
plt.show()
