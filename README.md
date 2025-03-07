# Fuzzy ART

A high-performance implementation of the Fuzzy Adaptive Resonance Theory (ART) algorithm with hardware acceleration support.

## Features

- **Unsupervised Learning**: Efficiently learns patterns with a single data pass
- **Online Learning**: Enables simultaneous learning and inference without retraining
- **Stability/Plasticity**: Preserves previously learned information (no catastrophic forgetting)

## Performance

- **Hardware-Accelerated Computation**: Automatically detects and uses the fastest available SIMD instructions
  - **Apple Silicon**: Leverages the Accelerate framework on macOS/ARM64 platforms (M1/M2/M3 chips)
  - **x86 Processors**: Utilizes AVX-512 instructions on compatible CPUs
  - **Fallback Support**: Provides optimized generic implementation for all other systems
- **Parallel Processing**: Multi-threaded implementation scales with available CPU cores (using a fixed-size worker-pool)
  - Category activation calculations
  - Fuzzy intersection computation
- **Memory Efficiency**: Reuses pre-allocated resources to minimize garbage collection overhead

Training on the full MNIST dataset completes in 5-6 minutes on Apple M1 Pro / Xeon W-3265M. 

## Installation

```bash
go get -u github.com/oblq/art
```

## Basic Usage

```go
package main

import (
	"fmt"

	"github.com/oblq/art"
)

func main() {
	// Create a new Fuzzy ART model with:
	// - 5 input features
	// - 0.9 vigilance parameter (controls category granularity)
	// - 0.01 choice parameter (influences category competition)
	// - 1.0 learning rate (controls weight update speed)
	model := art.NewFuzzyART(5, 0.9, 0.01, 1)
	defer model.Close() // Release resources when done

	// Prepare an input sample (values should be normalized between 0-1)
	input := []float64{0.1, 0.2, 0.3, 0.4, 0.5}

	// Train the model with the sample
	_, categoryIndex := model.Fit(input)

	// Test the model (with learning disabled)
	resonance, predictedCategoryIndex := model.Predict(input, false)

	fmt.Printf("Matched: %t (Category: %d, Resonance: %.4f)\n",
		categoryIndex == predictedCategoryIndex,
		predictedCategoryIndex,
		resonance)
}
```

## MNIST Example

The repository includes a full example using the MNIST dataset.

### Downloading the Dataset

```bash
make get-mnist
```

This command downloads the MNIST training and test sets to the `testdata` directory.

### Running the Example

```bash
make run
```

This trains and tests the Fuzzy ART model on the MNIST dataset, automatically using the optimal hardware acceleration for your system.

## About Adaptive Resonance Theory

Adaptive Resonance Theory (ART) is a cognitive and neural theory developed by Stephen Grossberg and Gail Carpenter that explains how the brain autonomously learns to categorize, recognize, and predict objects and events in a changing environment.

For a comprehensive overview, refer to [this research paper](https://www.semanticscholar.org/paper/Adaptive-Resonance-Theory%3A-How-a-brain-learns-to-a-Grossberg/71bc18bcafe1f4909a97b0b17a522dffe306ee6a).