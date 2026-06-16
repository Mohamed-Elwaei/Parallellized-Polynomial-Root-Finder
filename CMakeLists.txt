cmake_minimum_required(VERSION 3.14)
project(polyroots LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Library: all the numerical machinery, no main().
# ---------------------------------------------------------------------------
add_library(polyroots
    src/polynomial.cpp
    src/root_bound.cpp
    src/winding.cpp
    src/newton.cpp
    src/solver.cpp
)
target_include_directories(polyroots PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(polyroots PRIVATE -O2 -Wall -Wextra)

# ---------------------------------------------------------------------------
# Demo executable (the human-facing smoke test).
# ---------------------------------------------------------------------------
add_executable(demo apps/demo.cpp)
target_link_libraries(demo PRIVATE polyroots)

# ---------------------------------------------------------------------------
# Tests via GoogleTest, pulled in with FetchContent (no manual install).
# Toggle with -DPOLYROOTS_BUILD_TESTS=OFF.
# ---------------------------------------------------------------------------
option(POLYROOTS_BUILD_TESTS "Build unit tests" ON)
if(POLYROOTS_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(polyroots_tests
        tests/test_polynomial.cpp
        tests/test_root_bound.cpp
        tests/test_winding.cpp
        tests/test_newton.cpp
        tests/test_solver.cpp
    )
    target_include_directories(polyroots_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(polyroots_tests PRIVATE polyroots GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(polyroots_tests)
endif()
