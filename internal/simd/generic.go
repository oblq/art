package simd

import "math"

// generic implements Provider with standard Go code
type generic struct{}

func (p *generic) FuzzyIntersectionNorm(A, w []float64, fuzzyIntersectionOut []float64) (fiNorm, wNorm float64) {
	for i := 0; i < len(A); i++ {
		fuzzyIntersectionOut[i] = math.Min(A[i], w[i])
		fiNorm += fuzzyIntersectionOut[i]
		wNorm += w[i]
	}
	return
}

func (p *generic) SumFloat64(arr []float64) float64 {
	var sum float64
	for i := 0; i < len(arr); i++ {
		sum += arr[i]
	}
	return sum
}
