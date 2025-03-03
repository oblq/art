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

// accelerate implements Provider with Apple's Accelerate framework
type accelerate struct{}

func GetProvider() Provider {
	// todo: check if available
	return new(accelerate)
}

func (p *accelerate) FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	size := len(A)

	var intersectionPtr *C.double
	if intersection_out != nil {
		intersectionPtr = (*C.double)(&intersection_out[0])
	}

	sum := C.accelerate_fuzzy_intersection_float64(
		(C.size_t)(size),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		intersectionPtr,
	)

	return float64(sum)
}

func (p *accelerate) SumFloat64(arr []float64) float64 {
	size := len(arr)

	sum := C.accelerate_sum_float64(
		(C.size_t)(size),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}
