# Fuzzy ART

This repository contains a highly optimized implementation of the Fuzzy ART algorithm, part of the Adaptive Resonance Theory (ART) algorithms family developed by Stephen Grossberg and Gail Carpenter.

## Features
- **Unsupervised Learning**: Efficient learning with a single pass.
- **Explainable Results**: Clear and interpretable outcomes.
- **Stable Online Learning**: Incremental learning without the need to retrain from scratch, allowing simultaneous learning and inference.
- **No Catastrophic Forgetting**: Maintains previously learned information.

## Why Go?
Existing Python implementations of Fuzzy ART require significant time to complete training sessions on large datasets.

More than 4 hours are necessary to complete a training session on the full MNIST dataset, even in single thread this code takes an hour less, in parallel _**completes the training in 16 minutes on a MacBook Pro M1 Pro and in less than 9 minutes on a 48-thread Xeon W-3265M.**_

MacBook Pro M1 Pro: 10 cores, 16 threads, 16GB RAM:
![](./resources/MacbookPro_M1_Pro.png)

## Performance Optimization
This implementation utilizes:
- A worker pool sized to the number of available threads for parallelizing training and inference.
- Pre-allocated slices that are rotated and reused at every iteration.

Further optimizations may be possible. Detailed profiling of the code is planned for future enhancements, but current results are satisfactory.

## Installation

```bash
go get github.com/oblq/art
```

## Usage

```go
package main

import (
    "fmt"
    "github.com/oblq/art"
)

func main() {
    // Create a new Fuzzy ART model
    model := art.NewFuzzyART(5, 0.9, 0.00000001, 1)

    // Train the model with a sample
    category, categoryIndex := model.Train([]float64{0.1, 0.2, 0.3, 0.4, 0.5})

    // Test the model with a sample
    inferredCategory, inferredCategoryIndex := model.Infer([]float64{0.1, 0.2, 0.3, 0.4, 0.5}, false)

    fmt.Printf("Matched: %t\n", categoryIndex == inferredCategoryIndex)
```

## Run the example

If you cloned the project you can run the included example.

### Download MNIST Dataset
Run the following makefile target to download the MNIST dataset:
```bash
make get-mnist
```
This will place the training and test sets in the `example` folder.

### Configure Training
Open the `example/main.go` file and edit the following constants to test a subset of the dataset or leave them as `-1` to run the complete dataset:
```go
TRAIN_SAMPLES_PER_DIGIT = -1
TEST_SAMPLES_PER_DIGIT = -1
```

### Run Example
Execute the following command to start the training and test processes:
```bash
make run-example
```

## Adaptive Resonance Theory (ART)

Adaptive Resonance Theory, or ART, is a cognitive and neural theory of how the brain autonomously learns to categorize, recognize, and predict objects and events in a changing world pioneered by Stephen Grossberg and Gail Carpenter.

You can read all about it in this paper [here](https://www.semanticscholar.org/paper/Adaptive-Resonance-Theory%3A-How-a-brain-learns-to-a-Grossberg/71bc18bcafe1f4909a97b0b17a522dffe306ee6a?p2df).
