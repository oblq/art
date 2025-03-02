package simd

import "math"

// genericProvider implements Provider with standard Go code
type genericProvider struct{}

func (p *genericProvider) FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	var sum float64
	for i := 0; i < len(A); i++ {
		minimum := math.Min(A[i], w[i])
		if intersection_out != nil {
			intersection_out[i] = minimum
		}
		sum += minimum
	}
	return sum
}

func (p *genericProvider) SumFloat64(arr []float64) float64 {
	var sum float64
	for i := 0; i < len(arr); i++ {
		sum += arr[i]
	}
	return sum
}
