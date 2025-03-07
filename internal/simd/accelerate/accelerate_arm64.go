//go:build darwin && arm64

package accelerate

import "unsafe"

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

double accelerate_sum(const size_t n, double *arr) {
    double sum = 0.0;
    vDSP_sveD(arr, 1, &sum, n);

    return sum;
}

// update_fuzzy_weights updates weights using the formula:
// weights[i] = beta * fi[i] + (1-beta) * weights[i]
void update_fuzzy_weights(double* weights, const double* fi, double beta, int length) {
    // Use Apple's Accelerate framework for optimized vector operations on macOS

    // Compute beta * fi into a temporary buffer or in-place
    // Since vDSP operates on all elements at once, we need these operations

    // Step 1: Compute beta * fi
    vDSP_vsmulD(fi, 1, &beta, weights, 1, length);

    // Step 2: Compute (1-beta) * weights and add to the result
    double oneminusbeta = 1.0 - beta;
    vDSP_vsmaD(weights, 1, &oneminusbeta, weights, 1, weights, 1, length);
}
*/
import "C"

// Accelerate implements Provider with Apple's Accelerate framework
type Accelerate struct{}

func (p *Accelerate) FuzzyIntersectionNorm(A, w []float64, fuzzyIntersectionOut []float64) (float64, float64) {
	var wNormOut C.double
	fiNormOut := C.accelerate_fuzzy_intersection_norm(
		(C.size_t)(len(A)),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		(*C.double)(&fuzzyIntersectionOut[0]),
		&wNormOut,
	)

	return float64(fiNormOut), float64(wNormOut)
}

func (p *Accelerate) SumFloat64(arr []float64) float64 {
	sum := C.accelerate_sum(
		(C.size_t)(len(arr)),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}

func (p *Accelerate) UpdateFuzzyWeights(weights []float64, fi []float64, beta float64) {
	weightsPtr := (*C.double)(unsafe.Pointer(&weights[0]))
	fiPtr := (*C.double)(unsafe.Pointer(&fi[0]))
	C.update_fuzzy_weights(weightsPtr, fiPtr, C.double(beta), C.int(len(weights)))
}
