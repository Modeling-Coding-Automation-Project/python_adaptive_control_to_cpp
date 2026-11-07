// Exercise a realistic include order: another library has already instantiated
// ordinary backend norms before adaptive control installs its private adapter.
#include "python_numpy.hpp"
#include "python_optimization_panoc.hpp"
template struct PythonNumpy::NormalizationOperation::Normalizer<
    PythonNumpy::DenseMatrix_Type<double, 2, 1>, std::false_type>;
template struct PythonNumpy::NormalizationOperation::Normalizer<
    PythonNumpy::DenseMatrix_Type<float, 2, 1>, std::false_type>;

#include "data_driven_test_data.hpp"
#include "MCAP_tester.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace {
bool count_allocations = false;
std::size_t allocation_count = 0;
}

// The direct solvers are intended for fixed-memory embedded deployment.
void* operator new(std::size_t size) {
    if (count_allocations) ++allocation_count;
    void* memory = std::malloc(size ? size : 1);
    if (!memory) throw std::bad_alloc();
    return memory;
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }

namespace {
using namespace PythonAdaptiveControl;
using namespace DataDrivenTest;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void check_filters() {
    Tester::MCAPTester<T> tester;
    const T tolerance = std::is_same<T, float>::value ? T(1e-6) : T(1e-13);
    const RationalFilter<T, 1> filter(
        std::array<T, 2>{{T(2), T(1)}}, std::array<T, 2>{{T(2), T(-1)}});
    const std::array<T, 6> impulse = {{T(1), T(0), T(0), T(0), T(0), T(0)}};
    std::array<T, 6> actual = {{}}, repeated = {{}};
    require(filter.apply(impulse, actual) == Status::SUCCESS, "filter status");
    const std::array<T, 6> expected = {{T(1), T(1), T(.5), T(.25), T(.125), T(.0625)}};
    tester.expect_near(actual, expected, tolerance, "normalized impulse response");
    require(filter.apply(impulse, repeated) == Status::SUCCESS, "repeat filter status");
    tester.expect_near(repeated, actual, T(0), "filter resets initial state on each apply");
    FilterState<T, 1> state(filter);
    for (std::size_t k=0; k<impulse.size(); ++k)
        require(state.step(impulse[k], repeated[k]) == Status::SUCCESS, "streaming filter status");
    tester.expect_near(repeated, expected, tolerance, "streaming/batch filtering agree");
    state.reset();
    T reset_output = T(0);
    require(state.step(T(1), reset_output) == Status::SUCCESS, "reset streaming status");
    tester.expect_near(reset_output, T(1), tolerance, "streaming reset restores zero state");

    require(model<T>().apply(impulse, actual) == Status::SUCCESS, "delayed filter status");
    tester.expect_near(actual[0], T(0), T(0), "strictly proper model preserves delay");
    tester.expect_near(actual[1], T(.25), tolerance, "strictly proper model first response");

    const RationalFilter<T, 1> singular(
        std::array<T, 2>{{T(1), T(0)}}, std::array<T, 2>{{T(0), T(1)}});
    require(singular.apply(impulse, actual) != Status::SUCCESS, "zero denominator rejected");
    std::array<T, 6> bad_input = impulse;
    bad_input[3] = std::numeric_limits<T>::quiet_NaN();
    require(filter.apply(bad_input, actual) != Status::SUCCESS, "nonfinite input rejected");
    tester.throw_error_if_test_failed();
}

template <typename T>
void check_direct_solvers() {
    Tester::MCAPTester<T> tester;
    const T tolerance = std::is_same<T, float>::value ? T(2e-4) : T(2e-10);
    std::array<T, samples> u = {{}}, y = {{}};
    experiment(u, y);
    SolverOptions<T> options;
    options.rank_tolerance = std::is_same<T, float>::value ? T(1e-5) : T(1e-10);
    const auto m = model<T>();
    const auto forward = forward_basis<T>();
    const auto inverse = inverse_basis<T>();

    allocation_count = 0;
    count_allocations = true;
    const auto vrft = vrft_least_squares(u, y, m, forward, options);
    const auto frit = frit_least_squares(u, y, m, inverse, options);
    count_allocations = false;
    require(allocation_count == 0, "least squares allocated heap memory");
    require(vrft.status == Status::SUCCESS, "VRFT direct status");
    require(frit.status == Status::SUCCESS, "FRIT inverse direct status");
    require(vrft.rank == 2 && frit.rank == 2, "identified rank");
    tester.expect_near(vrft.parameters,
        std::array<T, 2>{{T(7.0 / 12.0), T(2.5)}}, tolerance, "analytical PI gains");
    tester.expect_near(frit.parameters,
        std::array<T, 2>{{T(1.2), T(-1.2)}}, tolerance, "analytical inverse-controller coefficients");
    require(std::isfinite(vrft.cost) && std::isfinite(frit.cost), "finite direct cost");
    tester.expect_near(vrft.cost, T(0), tolerance * tolerance, "exact VRFT fit");
    tester.expect_near(frit.cost, T(0), tolerance * tolerance, "exact FRIT fit");

    options.discard_samples = 13;
    const auto trimmed = vrft_least_squares(u, y, m, forward, options);
    require(trimmed.status == Status::SUCCESS, "discard sample status");
    tester.expect_near(trimmed.parameters, vrft.parameters, tolerance,
        "filter state preserved before discard boundary");
    const RationalFilter<T, 1> weight(
        std::array<T, 2>{{T(.4), T(0)}}, std::array<T, 2>{{T(1), T(-.6)}});
    const auto weighted = vrft_least_squares(u,y,m,forward,options,weight);
    require(weighted.status == Status::SUCCESS, "weighted VRFT status");
    tester.expect_near(weighted.parameters, vrft.parameters, tolerance,
        "common prefilter preserves realizable controller after discard");

    // Ridge reference obtained by an independent two-variable normal equation.
    options.ridge = T(.03);
    const auto regression = vrft_regression(u, y, m, forward);
    require(regression.status == Status::SUCCESS, "ridge regression status");
    double a = double(options.ridge), b = 0, c = double(options.ridge), d = 0, e = 0;
    const double count = double(samples - options.discard_samples);
    for (std::size_t k = options.discard_samples; k < samples; ++k) {
        const double x0 = regression.columns[0][k], x1 = regression.columns[1][k];
        a += x0 * x0 / count;
        b += x0 * x1 / count;
        c += x1 * x1 / count;
        d += x0 * double(regression.target[k]) / count;
        e += x1 * double(regression.target[k]) / count;
    }
    const double determinant = a * c - b * b;
    const std::array<T, 2> ridge_expected = {{T((d*c-b*e)/determinant), T((a*e-b*d)/determinant)}};
    const auto ridge_fit = vrft_least_squares(u, y, m, forward, options);
    require(ridge_fit.status == Status::SUCCESS, "ridge solve status");
    tester.expect_near(ridge_fit.parameters, ridge_expected, tolerance, "mean-cost ridge normalization");

    const auto trajectory = closed_loop<T, 80>(vrft.parameters);
    double desired = 0.0;
    for (std::size_t k = 0; k < trajectory.size(); ++k) {
        if (k) desired = .25 + .75 * desired;
        tester.expect_near(trajectory[k], T(desired), tolerance, "held-out step response");
    }
    tester.throw_error_if_test_failed();
}

template <typename T>
void check_full_rank_ill_conditioning() {
    Tester::MCAPTester<T> tester;
    Regression<T, 2, 64> regression;
    regression.status = Status::SUCCESS;
    const T delta = std::is_same<T, float>::value ? T(.003) : T(1e-5);
    for (std::size_t k=0; k<64; ++k) {
        regression.columns[0][k] = T(1);
        regression.columns[1][k] = T(1) + delta * (T(k)/T(63)-T(.5));
        regression.target[k] = T(2)-regression.columns[1][k];
    }
    SolverOptions<T> options;
    options.rank_tolerance = std::is_same<T,float>::value ? T(1e-6) : T(1e-10);
    const auto fitted = solve_least_squares(regression,options);
    require(fitted.status==Status::SUCCESS && fitted.rank==2, "ill-conditioned full-rank QR accepted");
    tester.expect_near(fitted.parameters,std::array<T,2>{{T(2),T(-1)}},
        std::is_same<T,float>::value ? T(.003) : T(3e-9), "QR avoids squared conditioning");
    tester.throw_error_if_test_failed();
}

template <typename T>
void check_scalar_and_pid() {
    Tester::MCAPTester<T> tester;
    const bool single = std::is_same<T,float>::value;
    std::array<T,128> u={{}}, y={{}};
    experiment(u,y);
    const LinearControllerBasis<T,1,1> proportional(
        std::array<std::array<T,2>,1>{{{{T(1),T(0)}}}},
        std::array<T,2>{{T(1),T(0)}});
    const RationalFilter<T,1> proportional_model(
        std::array<T,2>{{T(0),T(.24)}},std::array<T,2>{{T(1),T(-.46)}});
    SolverOptions<T> options;
    options.tolerance=single ? T(1e-5) : T(1e-9);
    options.max_iterations=1000;
    const std::array<T,1> scalar_initial={{T(1.1)}},scalar_lower={{T(.1)}},scalar_upper={{T(2)}};
    const auto sv=vrft_least_squares(u,y,proportional_model,proportional,options);
    const auto sf=frit_least_squares(u,y,proportional_model,proportional,options);
    const auto svo=vrft_optimize(u,y,proportional_model,proportional,scalar_initial,scalar_lower,scalar_upper,options);
    const auto sfo=frit_optimize(u,y,proportional_model,proportional,scalar_initial,scalar_lower,scalar_upper,options);
    require(sv.status==Status::SUCCESS && sf.status==Status::SUCCESS &&
            svo.status==Status::SUCCESS && sfo.status==Status::SUCCESS,"scalar algorithms converge");
    const T scalar_tolerance=single ? T(.002) : T(2e-6);
    tester.expect_near(sv.parameters[0],T(.8),scalar_tolerance,"scalar VRFT LS gain");
    tester.expect_near(sf.parameters[0],T(1.25),scalar_tolerance,"scalar FRIT inverse gain");
    tester.expect_near(svo.parameters[0],T(.8),scalar_tolerance,"scalar VRFT PANOC gain");
    tester.expect_near(sfo.parameters[0],T(.8),scalar_tolerance,"scalar FRIT PANOC gain");

    // Desired model is G*C/(1+G*C), for Kp=.6, Ki=2, Kd=.01.
    // All three PID terms are nonzero and the controller has order two.
    const RationalFilter<T,3> pid_model(
        std::array<T,4>{{T(0),T(.27),T(-.24),T(.03)}},
        std::array<T,4>{{T(1),T(-1.43),T(.46),T(.03)}});
    const LinearControllerBasis<T,3,3> pid(
        std::array<std::array<T,4>,3>{{
            {{T(1),T(-1),T(0),T(0)}},
            {{T(.1),T(0),T(0),T(0)}},
            {{T(10),T(-20),T(10),T(0)}}}},
        std::array<T,4>{{T(1),T(-1),T(0),T(0)}});
    const LinearControllerBasis<T,3,3> inverse_pid(
        std::array<std::array<T,4>,3>{{
            {{T(1),T(0),T(0),T(0)}},
            {{T(0),T(1),T(0),T(0)}},
            {{T(0),T(0),T(1),T(0)}}}},
        std::array<T,4>{{T(.9),T(-.8),T(.1),T(0)}});
    const auto pv=vrft_least_squares(u,y,pid_model,pid,options);
    const auto pf=frit_least_squares(u,y,pid_model,inverse_pid,options);
    require(pv.status==Status::SUCCESS && pf.status==Status::SUCCESS,"three-parameter PID LS status");
    const std::array<T,3> pid_truth={{T(.6),T(2),T(.01)}};
    const T pid_tolerance=single ? T(.002) : T(2e-9);
    tester.expect_near(pv.parameters,pid_truth,pid_tolerance,"three-parameter PID identification");
    tester.expect_near(pf.parameters,std::array<T,3>{{T(1),T(-1),T(0)}},pid_tolerance,
        "three-parameter inverse-controller identification");
    const std::array<T,3> pid_initial={{T(.8),T(1.4),T(.02)}};
    const std::array<T,3> pid_lower={{T(.1),T(.1),T(0)}};
    const std::array<T,3> pid_upper={{T(2),T(4),T(.1)}};
    options.tolerance=single ? T(1e-4) : T(1e-8);
    const auto pvo=vrft_optimize(u,y,pid_model,pid,pid_initial,pid_lower,pid_upper,options);
    const auto pfo=frit_optimize(u,y,pid_model,pid,pid_initial,pid_lower,pid_upper,options);
    require(pvo.status==Status::SUCCESS && pfo.status==Status::SUCCESS,"three-parameter PID optimizers converge");
    tester.expect_near(pvo.parameters,pid_truth,single ? T(.04) : T(.001),"optimized PID VRFT gains");
    tester.expect_near(pfo.parameters,pid_truth,single ? T(.04) : T(.001),"optimized PID FRIT gains");
    tester.throw_error_if_test_failed();
}

template <typename T>
void check_failures() {
    std::array<T, samples> u = {{}}, y = {{}};
    experiment(u, y);
    const auto m = model<T>();
    const auto forward = forward_basis<T>();
    SolverOptions<T> options;
    const LinearControllerBasis<T, 2, 1> duplicate(
        std::array<std::array<T, 2>, 2>{{{{T(1), T(0)}}, {{T(1), T(0)}}}},
        std::array<T, 2>{{T(1), T(0)}});
    auto rank_failure = vrft_least_squares(u, y, m, duplicate, options);
    require(rank_failure.status == Status::RANK_DEFICIENT, "duplicate basis must report rank deficiency");
    options.rank_tolerance = T(1e-4);
    const LinearControllerBasis<T, 2, 1> ill_conditioned(
        std::array<std::array<T, 2>, 2>{{{{T(1), T(0)}}, {{T(1), T(1e-7)}}}},
        std::array<T, 2>{{T(1), T(0)}});
    require(vrft_least_squares(u, y, m, ill_conditioned, options).status == Status::RANK_DEFICIENT,
        "near-collinear basis detected at requested tolerance");
    options.ridge = T(.01);
    require(vrft_least_squares(u, y, m, duplicate, options).status == Status::SUCCESS,
        "ridge resolves duplicate-basis singularity");
    options = SolverOptions<T>();
    options.discard_samples = samples;
    require(vrft_least_squares(u, y, m, forward, options).status == Status::INVALID_ARGUMENT,
        "discarding entire record rejected");
    options = SolverOptions<T>();
    options.ridge = T(-1);
    require(vrft_least_squares(u, y, m, forward, options).status == Status::INVALID_ARGUMENT,
        "negative ridge rejected");
    options = SolverOptions<T>();
    std::array<T, samples> nonfinite = y;
    nonfinite[19] = std::numeric_limits<T>::infinity();
    require(vrft_least_squares(u, nonfinite, m, forward, options).status != Status::SUCCESS,
        "infinite measurement rejected");
    nonfinite[19] = std::numeric_limits<T>::quiet_NaN();
    require(frit_least_squares(u, nonfinite, m, inverse_basis<T>(), options).status != Status::SUCCESS,
        "NaN measurement rejected");
    const auto singular = frit_cost_gradient(u, y, m, forward,
        std::array<T, 2>{{T(0), T(0)}}, options);
    require(singular.status == Status::SINGULAR_INVERSE, "zero controller inverse rejected");
}

template <typename T>
void check_gradient_and_optimization() {
    Tester::MCAPTester<T> tester;
    const bool single = std::is_same<T, float>::value;
    const T gain_tolerance = single ? T(.015) : T(2e-4);
    std::array<T, samples> u = {{}}, y = {{}};
    experiment(u, y);
    const auto m = model<T>();
    const auto forward = forward_basis<T>();
    const std::array<T, 2> initial = {{T(.9), T(1.4)}};
    const std::array<T, 2> lower = {{T(.1), T(.1)}};
    const std::array<T, 2> upper = {{T(2), T(5)}};
    SolverOptions<T> options;
    options.tolerance = single ? T(1e-5) : T(1e-9);
    options.max_iterations = 1000;
    options.discard_samples = 7;
    options.ridge = T(.003);
    const auto evaluated = frit_cost_gradient(u, y, m, forward, initial, options);
    require(evaluated.status == Status::SUCCESS, "ordinary FRIT evaluation status");
    tester.expect_near(evaluated.cost,
        T(independent_frit_cost(u, y, initial, options.discard_samples, double(options.ridge))),
        single ? T(3e-7) : T(1e-12), "independent ordinary FRIT objective");
    const T h = single ? T(1e-3) : T(1e-5);
    for (std::size_t i = 0; i < 2; ++i) {
        auto plus = initial;
        auto minus = initial;
        plus[i] += h;
        minus[i] -= h;
        const double finite_difference =
            (independent_frit_cost(u,y,plus,options.discard_samples,double(options.ridge)) -
             independent_frit_cost(u,y,minus,options.discard_samples,double(options.ridge))) /
            double(plus[i]-minus[i]);
        tester.expect_near(evaluated.gradient[i], T(finite_difference),
            single ? T(2e-5) : T(1e-9), "analytic FRIT gradient versus independent finite difference");
    }
    options.ridge = T(0);
    options.discard_samples = 0;
    const auto vrft = vrft_optimize(u, y, m, forward, initial, lower, upper, options);
    const auto frit = frit_optimize(u, y, m, forward, initial, lower, upper, options);
    require(vrft.status == Status::SUCCESS, "VRFT optimizer converges");
    require(frit.status == Status::SUCCESS, "FRIT optimizer converges");
    const std::array<T, 2> truth = {{T(7.0 / 12.0), T(2.5)}};
    tester.expect_near(vrft.parameters, truth, gain_tolerance, "optimized VRFT analytical gains");
    tester.expect_near(frit.parameters, truth, gain_tolerance, "optimized FRIT analytical gains");
    require(std::isfinite(vrft.cost) && std::isfinite(frit.cost), "optimizer costs finite");

    // A genuine active bound: the unconstrained proportional gain exceeds 0.4.
    const std::array<T, 2> constrained_upper = {{T(.4), T(5)}};
    const std::array<T, 2> bounded_initial = {{T(.3), T(1.4)}};
    const auto bounded = vrft_optimize(u,y,m,forward,bounded_initial,lower,constrained_upper,options);
    require(bounded.status == Status::SUCCESS, "bounded VRFT converges");
    const auto regression = vrft_regression(u,y,m,forward);
    double numerator = 0, denominator = 0;
    for (std::size_t k=0; k<samples; ++k) {
        const double x0=regression.columns[0][k], x1=regression.columns[1][k];
        numerator += x1*(double(regression.target[k])-.4*x0);
        denominator += x1*x1;
    }
    tester.expect_near(bounded.parameters[0], T(.4), gain_tolerance, "active proportional upper bound");
    tester.expect_near(bounded.parameters[1], T(numerator/denominator), gain_tolerance,
        "independent constrained optimum");
    for (std::size_t i=0; i<2; ++i)
        require(bounded.parameters[i]>=lower[i] && bounded.parameters[i]<=constrained_upper[i],
            "optimized result respects bounds");

    options.max_iterations = 1;
    options.tolerance = T(1e-12);
    const auto limited = frit_optimize(u,y,m,forward,initial,lower,upper,options);
    require(limited.status == Status::ITERATION_LIMIT, "iteration exhaustion is not success");
    require(std::isfinite(limited.cost), "iteration-limit result remains finite");
    const std::array<T, 2> bad_upper = {{T(0), T(0)}};
    require(vrft_optimize(u,y,m,forward,initial,lower,bad_upper,options).status == Status::INVALID_ARGUMENT,
        "inverted bounds rejected");
    tester.throw_error_if_test_failed();
}

template <typename T> void run() {
    check_filters<T>();
    check_direct_solvers<T>();
    check_failures<T>();
    check_gradient_and_optimization<T>();
    check_full_rank_ill_conditioning<T>();
    check_scalar_and_pid<T>();
}
} // namespace

int main() {
    try {
        run<double>();
        run<float>();
        std::cout << "All data-driven unit tests passed (double and float).\n";
        return 0;
    } catch (const std::exception& error) {
        count_allocations = false;
        std::cerr << "Data-driven unit test failed: " << error.what() << '\n';
        return 1;
    }
}
