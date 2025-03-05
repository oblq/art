//go:build darwin && arm64

package simd

/*
#cgo CFLAGS: -O3
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

double accelerate_fuzzy_intersection_norm(const size_t n, double *A, double *w, double *fuzzy_intersection_out, double *w_norm_out) {
    // Compute min(A[i], w[i])
    vDSP_vminD(A, 1, w, 1, fuzzy_intersection_out, 1, n);

    // Sum the min values
    double intersection_norm = 0.0;
    vDSP_sveD(fuzzy_intersection_out, 1, &intersection_norm, n);

    // Sum the w values directly into w_sum_out
    vDSP_sveD(w, 1, w_norm_out, n);

    return intersection_norm;
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

func (p *accelerate) FuzzyIntersectionNorm(A, w []float64, fuzzyIntersectionOut []float64) (float64, float64) {
	var wNormOut C.double
	fiNormOut = C.accelerate_fuzzy_intersection_norm(
		(C.size_t)(len(A)),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		(*C.double)(&fuzzyIntersectionOut[0]),
		&wNormOut,
	)

	return float64(fiNormOut), float64(wNormOut)
}

func (p *accelerate) SumFloat64(arr []float64) float64 {
	sum := C.accelerate_sum_float64(
		(C.size_t)(len(arr)),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}
