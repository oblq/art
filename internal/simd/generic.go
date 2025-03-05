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

func (p *generic) TopKActivations(choices []float64, indices []int, k int) ([]float64, []int) {
	n := len(choices)
	if n == 0 || k <= 0 {
		return []float64{}, []int{}
	}

	if k > n {
		k = n
	}

	topValues := make([]float64, k)
	topIndices := make([]int, k)

	// Initialize with invalid values
	for i := range topValues {
		topValues[i] = math.Inf(-1)
		topIndices[i] = -1
	}

	// Stream through all activations, maintaining top k
	for i := 0; i < n; i++ {
		val := choices[i]
		idx := indices[i]

		// Find insertion position based on value and tie-breaking
		pos := k
		for j := 0; j < k; j++ {
			if val > topValues[j] ||
				(val == topValues[j] && idx < topIndices[j]) {
				pos = j
				break
			}
		}

		// If an insertion position was found, shift and insert
		if pos < k {
			// Shift elements down
			for j := k - 1; j > pos; j-- {
				topValues[j] = topValues[j-1]
				topIndices[j] = topIndices[j-1]
			}
			// Insert the new value
			topValues[pos] = val
			topIndices[pos] = idx
		}
	}

	return topValues, topIndices
}
