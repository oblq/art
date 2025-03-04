# Fuzzy ART Neural Network in C++ with AVX-512

This is a C++ implementation of the Fuzzy Adaptive Resonance Theory (ART) neural network with parallel processing and AVX-512 SIMD optimizations. The implementation is based on a conversion from a Go version of the algorithm.

## Features

- Complete implementation of Fuzzy ART algorithm
- Parallel processing using a thread pool
- SIMD acceleration with AVX-512 intrinsics
- Thread-safe activation object pool to minimize memory allocations
- Comprehensive error handling and boundary checks

## Algorithm Overview

Fuzzy ART is an unsupervised learning neural network that performs online clustering by adaptively creating categories based on input patterns. The key parameters are:

- **Vigilance parameter (rho)**: Controls category granularity (0.0 to 1.0)
- **Choice parameter (alpha)**: Influences category competition (> 0.0)
- **Learning rate (beta)**: Controls weight update speed (0.0 to 1.0)

## Building the Project

The project uses CMake as its build system:

```bash
mkdir build
cd build
cmake ..
make
```

## Usage Example

```cpp
#include "fuzzy_art.hpp"
#include <vector>
#include <iostream>

int main() {
    // Create a Fuzzy ART network with 16-dimensional input
    // Parameters: input dimension, rho, alpha, beta
    FuzzyART art(16, 0.8, 0.01, 1.0);
    
    // Create an input vector (values should be normalized in [0,1])
    std::vector<float> input = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                                0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2};
    
    // Train the network
    auto [weights, category] = art.fit(input);
    std::cout << "Sample assigned to category: " << category << std::endl;
    
    // Predict using the trained network
    auto [pred_weights, pred_category] = art.predict(input, false);
    std::cout << "Predicted category: " << pred_category << std::endl;
    
    return 0;
}
```

## Performance Considerations

- The code is optimized for AVX-512 but includes fallbacks for systems without this instruction set
- For best performance, input dimensions should be multiples of 16 (for AVX-512 alignment)
- The thread pool automatically scales to the number of available CPU cores

## License

This project is available under the MIT License.