//go:build !darwin && arm64

package simd

/*
#cgo CFLAGS: -O3
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>

// NEON implementation of fuzzy intersection for ARM64
double neon_fuzzy_intersection_float64(const size_t n, double *A, double *w, double *intersection_out) {
    const size_t step = 2; // Process 2 doubles at a time with NEON
    const size_t vector_size = n / step;

    float64x2_t sum_vec = vdupq_n_f64(0.0);

    // Process 2 doubles at a time
    for (size_t i = 0; i < vector_size; ++i) {
        float64x2_t a_vec = vld1q_f64(&A[i * step]);
        float64x2_t w_vec = vld1q_f64(&w[i * step]);

        // Compute min(A[i], w[j][i])
        float64x2_t min_vec = vminq_f64(a_vec, w_vec);

        // Store result in the output array if not NULL
        if (intersection_out != NULL) {
            vst1q_f64(&intersection_out[i * step], min_vec);
        }

        // Accumulate sum
        sum_vec = vaddq_f64(sum_vec, min_vec);
    }

    // Extract and sum the elements from the vector
    double sum_arr[2];
    vst1q_f64(sum_arr, sum_vec);
    double sum = sum_arr[0] + sum_arr[1];

    // Handle remaining elements
    for (size_t i = vector_size * step; i < n; ++i) {
        double min_val = (A[i] < w[i]) ? A[i] : w[i];
        if (intersection_out != NULL) {
            intersection_out[i] = min_val;
        }
        sum += min_val;
    }

    return sum;
}

// NEON implementation of sum for ARM64
double neon_sum_float64(const size_t n, double *arr) {
    const size_t step = 2; // Process 2 doubles at a time with NEON
    const size_t vector_size = n / step;

    float64x2_t sum_vec = vdupq_n_f64(0.0);

    // Process 2 doubles at a time
    for (size_t i = 0; i < vector_size; ++i) {
        float64x2_t arr_vec = vld1q_f64(&arr[i * step]);
        sum_vec = vaddq_f64(sum_vec, arr_vec);
    }

    // Extract and sum the elements from the vector
    double sum_arr[2];
    vst1q_f64(sum_arr, sum_vec);
    double sum = sum_arr[0] + sum_arr[1];

    // Handle remaining elements
    for (size_t i = vector_size * step; i < n; ++i) {
        sum += arr[i];
    }

    return sum;
}
*/
import "C"

// neonProvider implements Provider with ARM NEON instructions
type neonProvider struct{}

func GetProvider() Provider {
	return new(neonProvider)
}

func (p *neonProvider) FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	size := len(A)

	var intersectionPtr *C.double
	if intersection_out != nil {
		intersectionPtr = (*C.double)(&intersection_out[0])
	}

	sum := C.neon_fuzzy_intersection_float64(
		(C.size_t)(size),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		intersectionPtr,
	)

	return float64(sum)
}

func (p *neonProvider) SumFloat64(arr []float64) float64 {
	size := len(arr)

	sum := C.neon_sum_float64(
		(C.size_t)(size),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}
