// Machine-readable SIL driver. The Python runner supplies its own fixtures and
// compares these C++ results; no Python binding or generated expected values.
#include "../test_vs/data_driven_test_data.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace PythonAdaptiveControl;
using namespace DataDrivenTest;

void number(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite SIL output");
    std::cout << value;
}

template <std::size_t N>
void array(const std::array<double, N>& value, std::size_t start = 0) {
    std::cout << '[';
    for (std::size_t k = start; k < N; ++k) {
        if (k != start) std::cout << ',';
        number(value[k]);
    }
    std::cout << ']';
}

void result(const FitResult<double, 2>& value) {
    if (value.status != Status::SUCCESS) throw std::runtime_error("SIL fit did not converge");
    std::cout << "{\"parameters\":";
    array(value.parameters);
    std::cout << ",\"cost\":";
    number(value.cost);
    std::cout << ",\"status\":\"" << status_name(value.status) << "\",\"rank\":"
              << value.rank << ",\"iterations\":" << value.iterations << '}';
}

template <std::size_t Order>
void run_case(const char* name, const RationalFilter<double, Order>& m, double noise, double ridge, std::size_t discard) {
    std::array<double, samples> u = {{}}, y = {{}};
    experiment(u, y, noise);
    std::array<std::array<double, Order + 1>, 2> forward_numerators = {{}};
    std::array<std::array<double, Order + 1>, 2> inverse_numerators = {{}};
    std::array<double, Order + 1> forward_denominator = {{}};
    std::array<double, Order + 1> inverse_denominator = {{}};
    forward_numerators[0][0] = 1.0; forward_numerators[0][1] = -1.0;
    forward_numerators[1][0] = .1;
    forward_denominator[0] = 1.0; forward_denominator[1] = -1.0;
    inverse_numerators[0][0] = 1.0; inverse_numerators[1][1] = 1.0;
    inverse_denominator[0] = 1.0; inverse_denominator[1] = -.7;
    const LinearControllerBasis<double, 2, Order> forward(forward_numerators, forward_denominator);
    const LinearControllerBasis<double, 2, Order> inverse(inverse_numerators, inverse_denominator);
    const std::array<double, 2> initial = {{.9, 1.4}};
    const std::array<double, 2> lower = {{.1, .1}};
    const std::array<double, 2> upper = {{2.0, 5.0}};
    SolverOptions<double> options;
    options.ridge = ridge;
    options.discard_samples = discard;
    options.tolerance = 1e-9;
    options.max_iterations = 1000;
    const auto vl = vrft_least_squares(u,y,m,forward,options);
    const auto vo = vrft_optimize(u,y,m,forward,initial,lower,upper,options);
    const auto fl = frit_least_squares(u,y,m,inverse,options);
    const auto fo = frit_optimize(u,y,m,forward,initial,lower,upper,options);
    const auto vr = vrft_regression(u,y,m,forward);
    const auto fr = frit_regression(u,y,m,inverse);
    const auto objective = frit_cost_gradient(u,y,m,forward,initial,options);
    if (vr.status != Status::SUCCESS || fr.status != Status::SUCCESS || objective.status != Status::SUCCESS)
        throw std::runtime_error("SIL regression/evaluation failed");

    std::cout << "{\"name\":\"" << name << "\",\"methods\":{\"vrft_least_squares\":";
    result(vl);
    std::cout << ",\"vrft_optimize\":";
    result(vo);
    std::cout << ",\"frit_least_squares\":";
    result(fl);
    std::cout << ",\"frit_optimize\":";
    result(fo);
    std::cout << "},\"vrft_columns\":[";
    array(vr.columns[0], discard); std::cout << ','; array(vr.columns[1], discard);
    std::cout << "],\"vrft_target\":"; array(vr.target, discard);
    std::cout << ",\"frit_columns\":[";
    array(fr.columns[0], discard); std::cout << ','; array(fr.columns[1], discard);
    std::cout << "],\"frit_target\":"; array(fr.target, discard);
    std::cout << ",\"frit_initial_cost\":"; number(objective.cost);
    std::cout << ",\"frit_initial_gradient\":"; array(objective.gradient);
    std::cout << ",\"vrft_closed_loop\":"; array(closed_loop<double, 64>(vo.parameters));
    std::cout << ",\"frit_closed_loop\":"; array(closed_loop<double, 64>(fo.parameters));
    std::cout << '}';
}
} // namespace

int main() {
    try {
        std::cout << std::setprecision(17) << "{\"cases\":[";
        run_case("exact", model<double>(), 0.0, 0.0, 0);
        std::cout << ',';
        run_case("noisy_ridge_discard", model<double>(), .003, .0003, 11);
        std::cout << ',';
        const RationalFilter<double, 2> delayed(
            std::array<double, 3>{{0.0, 0.0, .25}},
            std::array<double, 3>{{1.0, -.75, 0.0}});
        run_case("two_sample_delay", delayed, .001, .0001, 9);
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SIL failed: " << error.what() << '\n';
        return 1;
    }
}
