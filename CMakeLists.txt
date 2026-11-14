cmake_minimum_required(VERSION 3.16)
project(python_adaptive_control_to_cpp VERSION 0.1.0 LANGUAGES CXX)

option(ADAPTIVE_CONTROL_BUILD_TESTS "Build C++ unit tests" ON)
option(ADAPTIVE_CONTROL_BUILD_SAMPLES "Build VRFT and FRIT examples" ON)
option(ADAPTIVE_CONTROL_SANITIZERS "Enable address/undefined sanitizers" OFF)

add_library(python_adaptive_control INTERFACE)
add_library(MCAP::AdaptiveControl ALIAS python_adaptive_control)
target_compile_features(python_adaptive_control INTERFACE cxx_std_11)
target_include_directories(python_adaptive_control INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/data_driven)
target_include_directories(python_adaptive_control SYSTEM INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/base_utility_cpp/base_utility
    ${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/python_math_to_cpp/base_math
    ${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/python_math_to_cpp/python_math
    ${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/python_numpy_to_cpp/base_matrix
    ${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/python_numpy_to_cpp/python_numpy
    ${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/python_control_to_cpp/python_control
    ${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/python_optimization_to_cpp/python_optimization
    ${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/python_optimization_to_cpp/optimization_utility)

function(adaptive_control_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE MCAP::AdaptiveControl)
    set_target_properties(${name} PROPERTIES
        CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
        if(ADAPTIVE_CONTROL_SANITIZERS)
            target_compile_options(${name} PRIVATE
                -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${name} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

if(ADAPTIVE_CONTROL_BUILD_TESTS)
    enable_testing()
    adaptive_control_executable(data_driven_unit_tests test_vs/test_data_driven.cpp)
    target_include_directories(data_driven_unit_tests SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test_vs/MCAP_tester/include)
    add_test(NAME data_driven_unit_tests COMMAND data_driven_unit_tests)
endif()

if(ADAPTIVE_CONTROL_BUILD_SAMPLES)
    adaptive_control_executable(vrft_demo sample/data_driven/vrft_demo.cpp)
    adaptive_control_executable(frit_demo sample/data_driven/frit_demo.cpp)
    if(ADAPTIVE_CONTROL_BUILD_TESTS)
        add_test(NAME vrft_demo COMMAND vrft_demo)
        add_test(NAME frit_demo COMMAND frit_demo)
        # Samples print their workflow; numerical comparisons belong in SIL.
        set_tests_properties(vrft_demo PROPERTIES
            PASS_REGULAR_EXPRESSION "VRFT simulation complete."
            FAIL_REGULAR_EXPRESSION "ERROR:")
        set_tests_properties(frit_demo PROPERTIES
            PASS_REGULAR_EXPRESSION "FRIT simulation complete."
            FAIL_REGULAR_EXPRESSION "ERROR:")
    endif()
endif()
