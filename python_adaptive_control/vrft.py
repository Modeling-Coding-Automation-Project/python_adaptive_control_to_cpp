import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import control


class VRFT:
    def __init__(self, delta_time, reference_model, step_size):
        self.delta_time = delta_time
        self.reference_model = reference_model

        # structure of discrete pid controller
        self.beta = control.ss([[1.0, 0.0], [0.0, 0.0]],
                               [[1.0], [-1.0]],
                               [[0.0, 0.0], [self.delta_time, 0.0],
                                [0.0, -1.0 / self.delta_time]],
                               [[1.0], [0.0], [1.0 / self.delta_time]],
                               self.delta_time)

        self.step_size = step_size

        # Prefilter
        # Approxiate that the prefilter is the same as reference model.
        L = self.reference_model
        self.L_ss = control.ss(L)

    def simulate_prefilter(self, input_signal):
        ul = []
        X0 = np.zeros((self.L_ss.A.shape[0], 1))

        for i in range(self.step_size):
            # filer input
            u = np.array([[input_signal[i]]])

            # filer output
            y = self.L_ss.C @ X0 + self.L_ss.D @ u
            X0 = self.L_ss.A @ X0 + self.L_ss.B @ u

            ul.append(y)

        return np.array(ul).squeeze()

    def simulate_pseudo_error(self, output_signal):
        y_filter = []
        X0 = np.zeros((self.L_ss.A.shape[0], 1))

        for i in range(self.step_size):
            # plant input
            u = np.array([[output_signal[i]]])

            # plant output
            y = self.L_ss.C @ X0 + self.L_ss.D @ u
            X0 = self.L_ss.A @ X0 + self.L_ss.B @ u

            y_filter.append(y)

        y_filter = np.array(y_filter).squeeze()

        return np.array(output_signal - y_filter).squeeze()

    def simulate_controller_structure(self, input_signal):
        phi = []
        X0 = np.zeros((self.beta.A.shape[0], 1))

        for i in range(self.step_size):
            # controller input
            u = np.array([[input_signal[i]]])

            # controller output
            y = self.beta.C @ X0 + self.beta.D @ u
            X0 = self.beta.A @ X0 + self.beta.B @ u

            phi.append(y)

        return np.array(phi).squeeze()

    def estimate_gains(self, phi, ul):
        ul = ul.reshape(-1, 1)

        solution = np.linalg.lstsq(phi, ul, rcond=None)
        rho = solution[0]

        return rho

    def solve(self, input_signal, output_signal):
        # simulate prefilter
        ul = self.simulate_prefilter(input_signal)

        # simulate pseudo-error
        el = self.simulate_pseudo_error(output_signal)

        # simulate controller structure
        phi = self.simulate_controller_structure(el)

        rho = self.estimate_gains(phi, ul)

        return rho
