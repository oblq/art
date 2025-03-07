package simd

import (
	"fmt"
	"runtime"
)

// Provider defines the interface for platform-specific SIMD operations
type Provider interface {
	// FuzzyIntersectionNorm computes element-wise min between vectors and returns norms
	FuzzyIntersectionNorm(A, w []float64, fuzzyIntersectionOut []float64) (fiNorm float64, wNorm float64)

	// SumFloat64 computes the sum of all elements in an array
	SumFloat64(arr []float64) float64

	// UpdateFuzzyWeights updates weights according to the ART learning rule
	UpdateFuzzyWeights(W, fi []float64, beta float64)
}

var Shared Provider

func init() {
	Shared = GetProvider()
	if Shared == nil {
		Shared = new(generic)
	}

	fmt.Printf("Using %T on %s/%s\n", Shared, runtime.GOOS, runtime.GOARCH)
}
