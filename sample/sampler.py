import numpy as np
from scipy.interpolate import interp1d


class Sampler:
    @staticmethod
    def modify_same_sample_time_value(data, interval):
        if data[0, 0] == data[1, 0]:
            data = data[1:, :]

        for i, val in enumerate(data):
            if i == 0:
                data_out = data[0, :]
            else:
                if (i >= 2) and (data[i, 0] == data[i - 1, 0]):

                    time_dif = data[i, 0] - (data[i - 1, 0] +
                                             data[i - 2, 0]) / 2.0

                    if time_dif >= interval:
                        time_dif = interval

                    data_out[i - 1, 0] = data[i - 1, 0] - time_dif

                data_out = np.vstack((data_out, data[i, :]))

        return data_out

    @staticmethod
    def create_periodical(data, start_time, end_time, sampling_interval):

        data_inter = Sampler.modify_same_sample_time_value(
            data, sampling_interval)

        time_points = data_inter[:, 0]
        value_points = data_inter[:, 1]

        interp_func = interp1d(time_points, value_points,
                               kind='linear', fill_value='extrapolate')

        sample_times = np.arange(start_time, end_time, sampling_interval)
        sample_times = np.append(sample_times, end_time)

        sampled_values = interp_func(sample_times).reshape(-1, 1)

        sample_times = sample_times.reshape(-1, 1)

        return sample_times, sampled_values


class PulseGenerator:
    @staticmethod
    def generate_pulse_points(start_time, period, pulse_width, pulse_amplitude, duration):
        """
        Generate pulse points for a pulse signal.

        Parameters:
            start_time (float): The start time of the pulse [s].
            period (float): The period of the pulse [s].
            pulse_width (float): The width of the pulse as a percentage of the period [%].
            pulse_amplitude (float): The amplitude of the pulse.
            duration (float): Total duration of the signal [s].

        Returns:
            np.ndarray: Array of input points with columns [time, value].
        """
        time_points = []
        value_points = []

        current_time = start_time
        while current_time <= duration:
            # Pulse ON
            time_points.append(current_time)
            value_points.append(0.0)
            time_points.append(current_time)
            value_points.append(pulse_amplitude)

            pulse_on_duration = period * (pulse_width / 100.0)
            current_time += pulse_on_duration

            if current_time > duration:
                break

            # Pulse OFF
            time_points.append(current_time)
            value_points.append(pulse_amplitude)
            time_points.append(current_time)
            value_points.append(0.0)

            pulse_off_duration = period - pulse_on_duration
            current_time += pulse_off_duration

        if time_points[-1] < duration:
            time_points.append(duration)
            value_points.append(value_points[-1])

        return np.array(list(zip(time_points, value_points)))
