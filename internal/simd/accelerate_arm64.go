//go:build darwin && arm64

package simd

/*
#cgo CFLAGS: -O3
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

double accelerate_fuzzy_intersection_float64(const size_t n, double *A, double *w, double *intersection_out) {
    // Compute min(A[i], w[i])
    vDSP_vminD(A, 1, w, 1, intersection_out, 1, n);

    // Sum the min values
    double sum = 0.0;
    vDSP_sveD(intersection_out, 1, &sum, n);

    return sum;
}

double accelerate_sum_float64(const size_t n, double *arr) {
    double sum = 0.0;
    vDSP_sveD(arr, 1, &sum, n);

    return sum;
}
*/
import "C"
import "math"

// accelerate implements Provider with Apple's Accelerate framework
type accelerate struct{}

func GetProvider() Provider {
	// todo: check if available
	return new(accelerate)
}

func (p *accelerate) FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	var intersectionPtr = (*C.double)(&intersection_out[0])
	sum := C.accelerate_fuzzy_intersection_float64(
		(C.size_t)(len(A)),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		intersectionPtr,
	)

	return float64(sum)
}

func (p *accelerate) SumFloat64(arr []float64) float64 {
	sum := C.accelerate_sum_float64(
		(C.size_t)(len(arr)),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}

func (p *accelerate) TopKActivations(choices []float64, indices []int, k int) ([]float64, []int) {
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
