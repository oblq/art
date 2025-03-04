# Go Wrapper for C++ FuzzyART with AVX-512

This project provides a Go wrapper for the C++ implementation of the Fuzzy Adaptive Resonance Theory (ART) neural network with AVX-512 optimizations. The wrapper uses CGo to interface with the C++ code, allowing you to get the performance benefits of the optimized C++ implementation while using Go for your application logic.

## Features

- **High Performance**: Uses AVX-512 SIMD intrinsics for vectorized computation
- **Parallel Processing**: Leverages multi-threading for faster category activation calculations
- **Idiomatic Go Interface**: Simple and familiar Go API that handles all the C++ interaction details
- **Memory Safety**: Automatic cleanup of C++ resources through Go's garbage collection

## Building the Project

You need both Go and a C++ compiler (with AVX-512 support) installed on your system.

```bash
# Build the shared library and Go executable
make build

# Run the example
make run

# Clean up
make clean
```

## Go API Usage

```go
import (
    "fmt"
    "./art"  // Import the art package containing the FuzzyART wrapper
)

func main() {
    // Create a new FuzzyART model
    // Parameters: inputDim, rho, alpha, beta
    art, err := art.NewFuzzyART(16, 0.8, 0.01, 1.0)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    defer art.Close() // Ensure C++ resources are freed

    // Create an input vector (values should be in [0,1])
    input := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                      0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
    
    // Train the network
    weights, category, err := art.Fit(input)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Sample assigned to category: %d\n", category)
    
    // Predict using the trained network
    // The second parameter controls whether the model should learn from this input
    weights, category, err = art.Predict(input, false)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Predicted category: %d\n", category)
}
```

## Performance Considerations

- For best performance, use input dimensions that are multiples of 16 (for AVX-512 alignment)
- The C++ implementation automatically detects the number of available CPU cores and scales accordingly
- The training process is parallelized, but the wrapper introduces some overhead due to data conversion between Go and C++

## Requirements

- Go 1.11 or higher
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- CPU with AVX-512 support (Intel Skylake-X, Cascade Lake, Ice Lake, or newer)

## Implementation Details

- The Go wrapper uses CGo to call the C interface defined in `fuzzy_art_wrapper.h`
- The C interface in turn calls the C++ implementation in `fuzzy_art.cpp`
- Memory management is handled by Go's garbage collector, which calls `FuzzyART_Free` when the Go object is collected
- Vectors are converted between Go `[]float64` and C++ `std::vector<float>`, which introduces some overhead

## Troubleshooting

If you encounter issues with the shared library not being found, ensure that the `LD_LIBRARY_PATH` environment variable includes the directory containing `libfuzzyart.so`:

```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
```