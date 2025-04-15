import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import inspect


class SubplotsInfo:
    def __init__(self, signal_name, shape,
                 column, row, x_sequence,
                 x_sequence_name, line_style,
                 marker, label):

        self.signal_name = signal_name
        self.shape = shape
        self.column = column
        self.row = row

        self.x_sequence = x_sequence
        self.x_sequence_name = x_sequence_name

        self.line_style = line_style
        self.label = label
        self.marker = marker


class Configuration:
    def __init__(self):
        self.subplots_shape = np.zeros((2, 1), dtype=int)
        self.subplots_signals_list = []

        self.window_width_base = 6
        self.window_height_base = 4

        self.window_width_each_subplot = 4
        self.window_height_each_subplot = 2


class SimulationPlotter:
    def __init__(self):
        self.configuration = Configuration()
        self.name_to_object_dictionary = {}

    def append(self, signal_object):
        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the object_name that matches the matrix_in value
        object_name = None
        for name, value in caller_locals.items():
            if value is signal_object:
                object_name = name
                break

            # %% append object
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_object)
        else:
            self.name_to_object_dictionary[object_name] = [signal_object]

    def append_name(self, signal_object, object_name):
        if object_name in self.name_to_object_dictionary:
            self.name_to_object_dictionary[object_name].append(signal_object)
        else:
            self.name_to_object_dictionary[object_name] = [signal_object]

    def append_sequence(self, signal_sequence_object):
        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the object_name that matches the matrix_in value
        object_name = None
        for name, value in caller_locals.items():
            if value is signal_sequence_object:
                object_name = name
                break

            # %% append object
        for i in range(len(signal_sequence_object)):
            if object_name in self.name_to_object_dictionary:
                self.name_to_object_dictionary[object_name].append(
                    signal_sequence_object[i].reshape(-1, 1))
            else:
                self.name_to_object_dictionary[object_name] = [
                    signal_sequence_object[i].reshape(-1, 1)]

    def append_sequence_name(self, signal_sequence_object, object_name):
        for i in range(len(signal_sequence_object)):
            if object_name in self.name_to_object_dictionary:
                self.name_to_object_dictionary[object_name].append(
                    signal_sequence_object[i].reshape(-1, 1))
            else:
                self.name_to_object_dictionary[object_name] = [
                    signal_sequence_object[i].reshape(-1, 1)]

    def assign(self, signal_name, position,
               column=0, row=0, x_sequence=None,
               line_style="-", marker="", label=""):

        x_sequence_name = ""
        if x_sequence is not None:
            # %% inspect arguments
            # Get the caller's frame
            frame = inspect.currentframe().f_back
            # Get the caller's local variables
            caller_locals = frame.f_locals
            # Find the object_name that matches the matrix_in value
            x_sequence_name = None
            for name, value in caller_locals.items():
                if value is x_sequence:
                    x_sequence_name = name
                    break

        if (x_sequence is not None) and isinstance(x_sequence, str):
            x_sequence_name = x_sequence
            x_sequence = self.name_to_object_dictionary[x_sequence]

        # %% assign object
        shape = np.array([[position[0]], [position[1]]], dtype=int)

        self.configuration.subplots_signals_list.append(
            SubplotsInfo(signal_name, shape,
                         column, row, x_sequence,
                         x_sequence_name, line_style,
                         marker, label))

    def assign_all(self, signal_name, position,
                   x_sequence=None,
                   line_style="-", marker="", label=""):

        col_size = eval(
            f"self.name_to_object_dictionary[\"{signal_name}\"][0].shape[0]")
        row_size = eval(
            f"self.name_to_object_dictionary[\"{signal_name}\"][0].shape[1]")

        if label == "":
            label = signal_name

        for i in range(col_size):
            for j in range(row_size):
                label_text = label + "_" + str(i) + "_" + str(j)
                self.assign(signal_name, position=position,
                            column=i, row=j,
                            x_sequence=x_sequence,
                            line_style=line_style, marker=marker,
                            label=label_text)

    def pre_plot(self, suptitle=""):

        subplots_signals_list = self.configuration.subplots_signals_list

        if len(subplots_signals_list) == 0:
            print("No subplots to show.")
            return

        shape = np.zeros((2, 1), dtype=int)
        for signal_info in subplots_signals_list:
            if shape[0, 0] < signal_info.shape[0, 0] + 1:
                shape[0, 0] = signal_info.shape[0, 0] + 1
            if shape[1, 0] < signal_info.shape[1, 0] + 1:
                shape[1, 0] = signal_info.shape[1, 0] + 1

        self.configuration.subplots_shape = shape

        figure_size = (self.configuration.window_width_base +
                       self.configuration.window_width_each_subplot *
                       (shape[1, 0] - 1),
                       self.configuration.window_height_base +
                       self.configuration.window_height_each_subplot * (shape[0, 0] - 1))

        fig, axs = plt.subplots(shape[0, 0], shape[1, 0], figsize=figure_size)
        fig.suptitle(suptitle)

        for signal_info in subplots_signals_list:
            signal_object_list = self.name_to_object_dictionary[signal_info.signal_name]

            steps = len(signal_object_list)

            x_sequence_signal = np.zeros((steps, 1))
            if signal_info.x_sequence is not None:
                for i in range(steps):
                    x_sequence_signal[i, 0] = signal_info.x_sequence[i]
            else:
                x_sequence_signal = np.arange(steps)

            signal = np.zeros((steps, 1))
            if isinstance(signal_object_list[0], np.ndarray):
                for i in range(steps):
                    signal[i, 0] = signal_object_list[i][signal_info.column,
                                                         signal_info.row]
            else:
                for i in range(steps):
                    signal[i, 0] = signal_object_list[i]

            if signal_info.label == "":
                label_name = signal_info.signal_name + \
                    f"[{signal_info.column}, {signal_info.row}]"
            else:
                label_name = signal_info.label

            if shape[0] == 1 and shape[1] == 1:
                ax = axs
            elif shape[0] == 1:
                ax = axs[signal_info.shape[1, 0]]
            elif shape[1] == 1:
                ax = axs[signal_info.shape[0, 0]]
            else:
                ax = axs[signal_info.shape[0, 0], signal_info.shape[1, 0]]

            ax.plot(x_sequence_signal, signal,
                    label=label_name,
                    linestyle=signal_info.line_style, marker=signal_info.marker)
            mplcursors.cursor(ax)
            ax.legend()
            ax.set_xlabel(signal_info.x_sequence_name)
            ax.grid(True)

    def plot(self, suptitle=""):
        self.pre_plot(suptitle)

        plt.show()
