#include "data_driven_SIL.hpp"

namespace py = pybind11;
using namespace DataDrivenSIL;
using Tuner = PythonAdaptiveControl::VRFT<Scalar, PARAMETER_COUNT, ORDER>;

// Each call invokes the C++ library; Python owns the sample workflow and checks
// the result after acquisition, offline tuning and every feedback step.
PYBIND11_MODULE(VrftSIL, module) {
  bind_common(module);
  py::class_<Tuner>(module, "VRFT", py::module_local())
      .def(py::init<const Filter &, const Basis &>())
      .def("fit_least_squares", &fit_least_squares<Tuner>, py::arg("u"),
           py::arg("y"), py::arg("options") = Options())
      .def("fit_iterative", &fit_iterative<Tuner>, py::arg("u"), py::arg("y"),
           py::arg("initial"), py::arg("lower"), py::arg("upper"),
           py::arg("options") = Options())
      .def("regression", &regression<Tuner>, py::arg("u"), py::arg("y"),
           py::arg("options") = Options());
}
