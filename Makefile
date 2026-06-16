# ===========================================================================
# Lightweight Makefile alternative to the CMake build.
#
#   make            build the library + demo
#   make run        build and run the demo
#   make test       build and run the GoogleTest suite (needs gtest installed:
#                   e.g. `sudo apt-get install libgtest-dev`, or use the
#                   CMake build which fetches gtest automatically)
#   make clean      remove build artifacts
#
# The CMake build (see CMakeLists.txt) is the recommended path because it
# fetches GoogleTest for you. This Makefile is here for quick, dependency-free
# builds of the library and demo.
# ===========================================================================

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
BUILD    := build

LIB_SRC  := $(wildcard src/*.cpp)
LIB_OBJ  := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SRC))

TEST_SRC := $(wildcard tests/*.cpp)

.PHONY: all run test clean

all: $(BUILD)/demo

# --- library objects -------------------------------------------------------
$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- demo ------------------------------------------------------------------
$(BUILD)/demo: apps/demo.cpp $(LIB_OBJ) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(BUILD)/demo
	./$(BUILD)/demo

# --- tests (requires a system GoogleTest) ----------------------------------
$(BUILD)/polyroots_tests: $(TEST_SRC) $(LIB_OBJ) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Itests $^ -o $@ -lgtest -lgtest_main -lpthread

test: $(BUILD)/polyroots_tests
	./$(BUILD)/polyroots_tests

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
