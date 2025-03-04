cmake_minimum_required(VERSION 3.10)
project(FuzzyART)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable AVX-512 for GCC or Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx512f")
elseif(MSVC)
    # Enable AVX-512 for MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
endif()

# Add source files
set(SOURCES
    fuzzy_art.cpp
    main.cpp
)

# Create executable
add_executable(fuzzy_art ${SOURCES})

# Add OpenMP support for additional parallelism
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fuzzy_art PUBLIC OpenMP::OpenMP_CXX)
endif()

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fuzzy_art PRIVATE -Wall -Wextra)
elseif(MSVC)
    target_compile_options(fuzzy_art PRIVATE /W4)
endif()

# Install target
install(TARGETS fuzzy_art DESTINATION bin)