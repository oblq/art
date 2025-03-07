package simd

import (
	"math"
)

// generic implements Provider using standard Go code without SIMD
type generic struct{}

// FuzzyIntersectionNorm computes elementwise min between activations and weights,
// and returns the sum of the result and sum of weights
func (p *generic) FuzzyIntersectionNorm(A, w []float64, fuzzyIntersectionOut []float64) (float64, float64) {
	var fiNorm, wNorm float64

	for i := range A {
		fuzzyIntersectionOut[i] = math.Min(A[i], w[i])
		fiNorm += fuzzyIntersectionOut[i]
		wNorm += w[i]
	}

	return fiNorm, wNorm
}

// SumFloat64 computes the sum of all elements in the array
func (p *generic) SumFloat64(arr []float64) float64 {
	var sum float64
	for _, v := range arr {
		sum += v
	}
	return sum
}

// UpdateFuzzyWeights updates the mean weights in the Euclidean ART
func (p *generic) UpdateFuzzyWeights(W, fi []float64, beta float64) {
	for i := range W {
		W[i] = beta*fi[i] + (1-beta)*W[i]
	}
}
