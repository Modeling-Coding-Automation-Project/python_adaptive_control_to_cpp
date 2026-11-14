#include "data_driven_SIL.hpp"

namespace py = pybind11;
using namespace DataDrivenSIL;
using Tuner = PythonAdaptiveControl::FRIT<Scalar, PARAMETER_COUNT, ORDER>;
using DirectTuner =
    PythonAdaptiveControl::FRITLeastSquares<Scalar, PARAMETER_COUNT, ORDER>;

namespace {
template <std::size_t N>
py::tuple evaluate(const Tuner &tuner, const Parameters &parameters,
                   const InputArray &u, const InputArray &y,
                   const Options &options) {
  const auto value = tuner.cost_gradient(parameters, read_vector<N>(u, "u"),
                                         read_vector<N>(y, "y"), options);
  require_success(value.status, "FRIT cost and gradient");
  return py::make_tuple(value.cost, numpy_vector(value.gradient));
}

py::tuple cost_gradient(const Tuner &tuner, const InputArray &parameters,
                        const InputArray &u, const InputArray &y,
                        const Options &options) {
  const Parameters values =
      read_vector<PARAMETER_COUNT>(parameters, "parameters");
  if (record_count(u, y) == 128)
    return evaluate<128>(tuner, values, u, y, options);
  return evaluate<256>(tuner, values, u, y, options);
}
} // namespace

PYBIND11_MODULE(FritSIL, module) {
  bind_common(module);
  py::class_<Tuner>(module, "FRIT", py::module_local())
      .def(py::init<const Filter &, const Basis &>())
      .def("fit_iterative", &fit_iterative<Tuner>, py::arg("u"), py::arg("y"),
           py::arg("initial"), py::arg("lower"), py::arg("upper"),
           py::arg("options") = Options())
      .def("cost_gradient", &cost_gradient, py::arg("parameters"), py::arg("u"),
           py::arg("y"), py::arg("options") = Options());
  py::class_<DirectTuner>(module, "FRITLeastSquares", py::module_local())
      .def(py::init<const Filter &, const Basis &>())
      .def("fit_least_squares", &fit_least_squares<DirectTuner>, py::arg("u"),
           py::arg("y"), py::arg("options") = Options())
      .def("regression", &regression<DirectTuner>, py::arg("u"), py::arg("y"),
           py::arg("options") = Options());
}
