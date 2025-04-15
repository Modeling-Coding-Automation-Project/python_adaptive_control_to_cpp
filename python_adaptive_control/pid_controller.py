import math
import numpy as np

MINIMUM_OUTPUT_DEFAULT = -np.inf
MAXIMUM_OUTPUT_DEFAULT = np.inf


def saturation(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class DiscretePID_Controller:
    def __init__(self, delta_time=0.0, Kp=0.0, Ki=0.0, Kd=0.0, N=100.0, Kb=0.0,
                 minimum_output=MINIMUM_OUTPUT_DEFAULT, maximum_output=MAXIMUM_OUTPUT_DEFAULT):
        # Public variables
        self.delta_time = delta_time
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.N = N  # Filter coefficient
        self.Kb = Kb  # Back calculation coefficient
        self.minimum_output = minimum_output
        self.maximum_output = maximum_output

        # Private variables
        self._integration_store = 0.0
        self._differentiation_store = 0.0
        self._PID_output = 0.0
        self._saturated_PID_output = 0.0

        if isinstance(self.Kp, float) and \
                isinstance(self.Ki, float) and \
                isinstance(self.Kd, float):

            self.data_type = np.float64.__name__
        else:
            raise TypeError("controller gains must be a float")

    def update(self, error):
        self._PID_output = (self._calculate_P_term(error) +
                            self._calculate_I_term(error) +
                            self._calculate_D_term(error))

        self._saturated_PID_output = saturation(
            self._PID_output, self.minimum_output, self.maximum_output)

        return self._saturated_PID_output

    def reset(self):
        self._integration_store = 0.0
        self._differentiation_store = 0.0

    def get_integration_store(self):
        return self._integration_store

    def get_differentiation_store(self):
        return self._differentiation_store

    def _back_calculation_for_I_term(self, error):
        return (self.Ki * error +
                self.Kb * (self._saturated_PID_output - self._PID_output))

    def _calculate_P_term(self, error):
        return self.Kp * error

    def _calculate_I_term(self, error):
        self._integration_store += (self._back_calculation_for_I_term(error) *
                                    self.delta_time)
        return self._integration_store

    def _calculate_D_term(self, error):
        # Incomplete differentiator
        output = self.N * ((self.Kd * error) - self._differentiation_store)

        self._differentiation_store += output * self.delta_time

        return output
