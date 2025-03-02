package simd

import (
	"fmt"
	"runtime"
)

// Provider defines the interface for platform-specific SIMD operations
type Provider interface {
	FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64
	SumFloat64(arr []float64) float64
}

var defaultProvider = getBestAvailableProvider()

// getBestAvailableProvider returns the best SIMD provider for the current platform
// or nil if no specific provider is available
func getBestAvailableProvider() Provider {
	return new(genericProvider)
	provider := GetProvider()
	if provider == nil {
		provider = new(genericProvider)
	}

	fmt.Printf("Using %T on %s/%s", provider, runtime.GOOS, runtime.GOARCH)
	return provider
}

// FuzzyIntersectionSum computes elementwise min between A and w and returns the sum
// This is the facade function that delegates to the appropriate implementation
func FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	return defaultProvider.FuzzyIntersectionSum(A, w, intersection_out)
}

// SumFloat64 computes the sum of all elements in the array
// This is the facade function that delegates to the appropriate implementation
func SumFloat64(arr []float64) float64 {
	return defaultProvider.SumFloat64(arr)
}
