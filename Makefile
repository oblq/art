.PHONY: all clean build run

# Variables
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mavx512f -fPIC
LDFLAGS = -shared -lstdc++
GO = go

# Source files
CPP_SOURCES = fuzzy_art.cpp fuzzy_art_wrapper.cpp
CPP_HEADERS = fuzzy_art.hpp fuzzy_art_wrapper.h
GO_SOURCES = fuzzy_art_wrapper_api.go example_usage.go

# Output files
LIB_NAME = libfuzzyart.so
EXECUTABLE = fuzzy_art_example

all: build

build: $(LIB_NAME) $(EXECUTABLE)

# Build the shared library
$(LIB_NAME): $(CPP_SOURCES) $(CPP_HEADERS)
	$(CXX) $(CXXFLAGS) $(CPP_SOURCES) -o $@ $(LDFLAGS)

# Build the Go executable
$(EXECUTABLE): $(LIB_NAME) $(GO_SOURCES)
	@# Set LD_LIBRARY_PATH to include the current directory
	@# This is needed for Go to find the shared library
	LD_LIBRARY_PATH=. $(GO) build -o $@ example_usage.go

run: build
	@# Run the executable with LD_LIBRARY_PATH set
	LD_LIBRARY_PATH=. ./$(EXECUTABLE)

clean:
	rm -f $(LIB_NAME) $(EXECUTABLE)