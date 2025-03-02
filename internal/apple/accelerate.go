package apple

/*
#cgo CFLAGS: -O3
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

double accelerate_fuzzy_intersection_float64(const size_t n, double *A, double *w, double *intersection_out) {
    // Create a temporary buffer if intersection_out is NULL
    double *min_buffer;
    double *temp_buffer = NULL;

    if (intersection_out != NULL) {
        min_buffer = intersection_out;
    } else {
        // Allocate temporary buffer
        temp_buffer = (double*)malloc(n * sizeof(double));
        if (!temp_buffer) return 0.0; // Handle allocation failure
        min_buffer = temp_buffer;
    }

    // Copy A to min_buffer
    memcpy(min_buffer, A, n * sizeof(double));

    // Compute min(A[i], w[i])
    vDSP_vminD(min_buffer, 1, w, 1, min_buffer, 1, n);

    // Sum the min values
    double sum = 0.0;
    vDSP_sveD(min_buffer, 1, &sum, n);

    // Free temporary buffer if allocated
    if (temp_buffer) {
        free(temp_buffer);
    }

    return sum;
}

double accelerate_sum_float64(const size_t n, double *arr) {
    double sum = 0.0;
    vDSP_sveD(arr, 1, &sum, n);
    return sum;
}
*/
import "C"

// FuzzyIntersectionSumAccelerate computes elementwise min between A and w using Accelerate
func FuzzyIntersectionSumAccelerate(A, w []float64, intersection_out []float64) float64 {
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

// SumFloat64Accelerate computes the sum of all elements in the array using Accelerate
func SumFloat64Accelerate(arr []float64) float64 {
	size := len(arr)

	sum := C.accelerate_sum_float64(
		(C.size_t)(size),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}
