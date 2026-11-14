#ifndef DATA_DRIVEN_SIL_HPP_
#define DATA_DRIVEN_SIL_HPP_

#include "data_driven.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <string>

// Conversion and bindings for SIL only. The control library remains independent
// of Python and continues to return Status rather than throwing exceptions.
namespace DataDrivenSIL {
namespace py = pybind11;
using Scalar = double;
constexpr std::size_t PARAMETER_COUNT = 2;
// Trailing zeros let one binding exercise the sample's first-order filters and
// the separate two-sample-delay regression case.
constexpr std::size_t ORDER = 2;
using InputArray =
    py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
using Filter = PythonAdaptiveControl::RationalFilter<Scalar, ORDER>;
using FilterState = PythonAdaptiveControl::FilterState<Scalar, ORDER>;
using Basis =
    PythonAdaptiveControl::LinearControllerBasis<Scalar, PARAMETER_COUNT,
                                                 ORDER>;
using Parameters = std::array<Scalar, PARAMETER_COUNT>;
using Options = PythonAdaptiveControl::SolverOptions<Scalar>;
using Fit = PythonAdaptiveControl::FitResult<Scalar, PARAMETER_COUNT>;
using Status = PythonAdaptiveControl::Status;

inline void require_success(Status status, const char *operation) {
  if (status != Status::SUCCESS)
    throw py::value_error(std::string(operation) + ": " +
                          PythonAdaptiveControl::status_name(status));
}

template <std::size_t N>
std::array<Scalar, N> read_vector(const InputArray &input, const char *name) {
  if (input.ndim() != 1 || input.shape(0) != static_cast<py::ssize_t>(N))
    throw py::value_error(std::string(name) + " has the wrong vector length");
  std::array<Scalar, N> result = {};
  const auto values = input.unchecked<1>();
  for (std::size_t index = 0; index < N; ++index)
    result[index] = values(static_cast<py::ssize_t>(index));
  return result;
}

template <std::size_t N>
py::array_t<Scalar> numpy_vector(const std::array<Scalar, N> &values) {
  py::array_t<Scalar> output(static_cast<py::ssize_t>(N));
  auto view = output.mutable_unchecked<1>();
  for (std::size_t index = 0; index < N; ++index)
    view(static_cast<py::ssize_t>(index)) = values[index];
  return output;
}

inline Filter::Coefficients read_coefficients(const InputArray &input,
                                              const char *name) {
  if (input.ndim() != 1 || input.shape(0) < 1 ||
      input.shape(0) > static_cast<py::ssize_t>(ORDER + 1))
    throw py::value_error(std::string(name) +
                          " must have one to three coefficients");
  Filter::Coefficients coefficients = {};
  const auto values = input.unchecked<1>();
  for (py::ssize_t index = 0; index < input.shape(0); ++index)
    coefficients[static_cast<std::size_t>(index)] = values(index);
  return coefficients;
}

inline Filter make_filter(const InputArray &numerator,
                          const InputArray &denominator) {
  const Filter filter(read_coefficients(numerator, "numerator"),
                      read_coefficients(denominator, "denominator"));
  require_success(filter.validate(), "filter construction");
  return filter;
}

inline Basis make_basis(const InputArray &numerators,
                        const InputArray &denominator) {
  if (numerators.ndim() != 2 ||
      numerators.shape(0) != static_cast<py::ssize_t>(PARAMETER_COUNT) ||
      numerators.shape(1) < 1 ||
      numerators.shape(1) > static_cast<py::ssize_t>(ORDER + 1))
    throw py::value_error("basis numerators must have shape (2, 1..3)");
  std::array<Filter::Coefficients, PARAMETER_COUNT> b = {};
  const auto values = numerators.unchecked<2>();
  for (std::size_t row = 0; row < PARAMETER_COUNT; ++row)
    for (py::ssize_t column = 0; column < numerators.shape(1); ++column)
      b[row][static_cast<std::size_t>(column)] =
          values(static_cast<py::ssize_t>(row), column);
  const Basis basis(b, read_coefficients(denominator, "basis denominator"));
  require_success(basis.validate(), "basis construction");
  return basis;
}

inline py::array_t<Scalar> filter_signal(const Filter &filter,
                                         const InputArray &input) {
  if (input.ndim() != 1 || input.shape(0) < 1)
    throw py::value_error("signal must be a nonempty vector");
  auto state = filter.state();
  require_success(state.status(), "filter state");
  py::array_t<Scalar> output(input.shape(0));
  const auto values = input.unchecked<1>();
  auto result = output.mutable_unchecked<1>();
  for (py::ssize_t index = 0; index < input.shape(0); ++index) {
    Scalar sample = 0.0;
    require_success(state.step(values(index), sample), "filtering");
    result(index) = sample;
  }
  return output;
}

inline std::size_t record_count(const InputArray &u, const InputArray &y) {
  if (u.ndim() != 1 || y.ndim() != 1 || u.shape(0) != y.shape(0))
    throw py::value_error("u and y must be vectors of the same length");
  if (u.shape(0) != 128 && u.shape(0) != 256)
    throw py::value_error(
        "these SIL bindings support 128 or 256 recorded samples");
  return static_cast<std::size_t>(u.shape(0));
}

template <typename Tuner>
Fit fit_least_squares(const Tuner &tuner, const InputArray &u,
                      const InputArray &y, const Options &options) {
  if (record_count(u, y) == 128)
    return tuner.fit_least_squares(read_vector<128>(u, "u"),
                                   read_vector<128>(y, "y"), options);
  return tuner.fit_least_squares(read_vector<256>(u, "u"),
                                 read_vector<256>(y, "y"), options);
}

template <typename Tuner>
Fit fit_iterative(const Tuner &tuner, const InputArray &u, const InputArray &y,
                  const InputArray &initial, const InputArray &lower,
                  const InputArray &upper, const Options &options) {
  const Parameters initial_values =
      read_vector<PARAMETER_COUNT>(initial, "initial");
  const Parameters lower_values = read_vector<PARAMETER_COUNT>(lower, "lower");
  const Parameters upper_values = read_vector<PARAMETER_COUNT>(upper, "upper");
  if (record_count(u, y) == 128)
    return tuner.fit_iterative(read_vector<128>(u, "u"),
                               read_vector<128>(y, "y"), initial_values,
                               lower_values, upper_values, options);
  return tuner.fit_iterative(read_vector<256>(u, "u"), read_vector<256>(y, "y"),
                             initial_values, lower_values, upper_values,
                             options);
}

template <std::size_t N, typename Tuner>
py::tuple regression_record(const Tuner &tuner, const InputArray &u,
                            const InputArray &y, const Options &options) {
  if (!PythonAdaptiveControl::detail::valid_options<Scalar, N>(options))
    throw py::value_error("invalid regression options");
  const auto regression =
      tuner.regression(read_vector<N>(u, "u"), read_vector<N>(y, "y"));
  require_success(regression.status, "regression");
  const py::ssize_t rows =
      static_cast<py::ssize_t>(N - options.discard_samples);
  py::array_t<Scalar> columns(
      {rows, static_cast<py::ssize_t>(PARAMETER_COUNT)});
  py::array_t<Scalar> target(rows);
  auto x = columns.mutable_unchecked<2>();
  auto t = target.mutable_unchecked<1>();
  for (py::ssize_t row = 0; row < rows; ++row) {
    const std::size_t index =
        static_cast<std::size_t>(row) + options.discard_samples;
    t(row) = regression.target[index];
    for (std::size_t column = 0; column < PARAMETER_COUNT; ++column)
      x(row, static_cast<py::ssize_t>(column)) =
          regression.columns[column][index];
  }
  return py::make_tuple(columns, target);
}

template <typename Tuner>
py::tuple regression(const Tuner &tuner, const InputArray &u,
                     const InputArray &y, const Options &options) {
  if (record_count(u, y) == 128)
    return regression_record<128>(tuner, u, y, options);
  return regression_record<256>(tuner, u, y, options);
}

inline void bind_common(py::module_ &module) {
  py::class_<Options>(module, "SolverOptions", py::module_local())
      .def(py::init<>())
      .def_readwrite("ridge", &Options::ridge)
      .def_readwrite("tolerance", &Options::tolerance)
      .def_readwrite("rank_tolerance", &Options::rank_tolerance)
      .def_readwrite("max_iterations", &Options::max_iterations)
      .def_readwrite("discard_samples", &Options::discard_samples);
  py::class_<Fit>(module, "FitResult", py::module_local())
      .def_property_readonly(
          "parameters",
          [](const Fit &fit) { return numpy_vector(fit.parameters); })
      .def_readonly("cost", &Fit::cost)
      .def_readonly("rank", &Fit::rank)
      .def_readonly("iterations", &Fit::iterations)
      .def_property_readonly("status",
                             [](const Fit &fit) {
                               return PythonAdaptiveControl::status_name(
                                   fit.status);
                             })
      .def_property_readonly("success", &Fit::has_converged);
  py::class_<FilterState>(module, "FilterState", py::module_local())
      .def("step",
           [](FilterState &state, Scalar input) {
             Scalar output = 0.0;
             require_success(state.step(input, output), "filter state step");
             return output;
           })
      .def("reset", &FilterState::reset);
  py::class_<Filter>(module, "RationalFilter", py::module_local())
      .def(py::init(&make_filter), py::arg("numerator"), py::arg("denominator"))
      .def("filter", &filter_signal)
      .def("state",
           [](const Filter &filter) {
             auto state = filter.state();
             require_success(state.status(), "filter state construction");
             return state;
           })
      .def("inverse", [](const Filter &filter) {
        Filter inverse;
        require_success(filter.inverse(inverse), "filter inversion");
        return inverse;
      });
  py::class_<Basis>(module, "LinearControllerBasis", py::module_local())
      .def(py::init(&make_basis), py::arg("numerators"), py::arg("denominator"))
      .def("controller", [](const Basis &basis, const InputArray &parameters) {
        const auto controller = basis.controller(
            read_vector<PARAMETER_COUNT>(parameters, "parameters"));
        require_success(controller.validate(), "controller construction");
        return controller;
      });
}
} // namespace DataDrivenSIL
#endif
