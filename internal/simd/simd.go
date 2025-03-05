package simd

import (
	"fmt"
	"runtime"
)

// Provider defines the interface for platform-specific SIMD operations
type Provider interface {
	FuzzyIntersectionNorm(A, w []float64, fuzzyIntersectionOut []float64) (fiNorm, wNorm float64)
	SumFloat64(arr []float64) float64
	TopKActivations(choices []float64, indices []int, k int) ([]float64, []int)
}

var defaultProvider = getBestAvailableProvider()

// getBestAvailableProvider returns the best SIMD provider for the current platform
// or nil if no specific provider is available
func getBestAvailableProvider() Provider {
	//return new(generic)

	provider := GetProvider()
	if provider == nil {
		provider = new(generic)
	}

	fmt.Printf("Using %T on %s/%s\n", provider, runtime.GOOS, runtime.GOARCH)
	return provider
}

// FuzzyIntersectionNorm computes elementwise min between A and w and returns the sum
// This is the facade function that delegates to the appropriate implementation
func FuzzyIntersectionNorm(A, w []float64, intersectionOut []float64) (float64, float64) {
	return defaultProvider.FuzzyIntersectionNorm(A, w, intersectionOut)
}

// SumFloat64 computes the sum of all elements in the array
// This is the facade function that delegates to the appropriate implementation
func SumFloat64(arr []float64) float64 {
	return defaultProvider.SumFloat64(arr)
}

// TopKActivations finds the top k activations and their indices.
// This is a wrapper function that dispatches to the appropriate provider.
func TopKActivations(choices []float64, indices []int, k int) ([]float64, []int) {
	return defaultProvider.TopKActivations(choices, indices, k)
}
