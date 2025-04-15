import numpy as np
from scipy.interpolate import interp1d


class Sampler:
    @staticmethod
    def modify_same_sample_time_value(data, interval):
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
