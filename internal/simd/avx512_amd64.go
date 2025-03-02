//go:build amd64

package simd

/*
#cgo CFLAGS: -mavx512f -mavx512vl -mavx512bw -mavx512dq
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>

// Computes the fuzzy intersection (elementwise min) between two arrays and returns the sum
double avx512_fuzzy_intersection_float64(const size_t n, double *A, double *w, double *intersection_out)
{
    static const size_t single_size = 8; // 8 doubles per AVX-512 register
    const size_t end = n / single_size;

    __m512d sum_vec = _mm512_setzero_pd();

    // Process 8 doubles at a time
    for(size_t i = 0; i < end; ++i) {
        __m512d a_vec = _mm512_loadu_pd(A + i * single_size);
        __m512d w_vec = _mm512_loadu_pd(w + i * single_size);

        // Compute min(A[i], w[j][i])
        __m512d min_vec = _mm512_min_pd(a_vec, w_vec);

        // Store result in the output array
        if (intersection_out != NULL) {
            _mm512_storeu_pd(intersection_out + i * single_size, min_vec);
        }

        // Accumulate sum
        sum_vec = _mm512_add_pd(sum_vec, min_vec);
    }

    // Reduce sum vector to a single value
    double sum = 0.0;
    double *sum_arr = (double*)&sum_vec;
    for (int i = 0; i < 8; i++) {
        sum += sum_arr[i];
    }

    // Handle remaining elements
    for(size_t i = end * single_size; i < n; ++i) {
        double min_val = A[i] < w[i] ? A[i] : w[i];
        if (intersection_out != NULL) {
            intersection_out[i] = min_val;
        }
        sum += min_val;
    }

    return sum;
}

// Computes the sum of an array using AVX-512
double avx512_sum_float64(const size_t n, double *arr)
{
    static const size_t single_size = 8; // 8 doubles per AVX-512 register
    const size_t end = n / single_size;

    __m512d sum_vec = _mm512_setzero_pd();

    // Process 8 doubles at a time
    for(size_t i = 0; i < end; ++i) {
        __m512d arr_vec = _mm512_loadu_pd(arr + i * single_size);
        sum_vec = _mm512_add_pd(sum_vec, arr_vec);
    }

    // Reduce sum vector to a single value
    double sum = 0.0;
    double *sum_arr = (double*)&sum_vec;
    for (int i = 0; i < 8; i++) {
        sum += sum_arr[i];
    }

    // Handle remaining elements
    for(size_t i = end * single_size; i < n; ++i) {
        sum += arr[i];
    }

    return sum;
}
*/
import "C"

// avx512Provider implements Provider with AVX-512 instructions
type avx512Provider struct{}

func (p *avx512Provider) FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	size := len(A)

	var intersectionPtr *C.double
	if intersection_out != nil {
		intersectionPtr = (*C.double)(&intersection_out[0])
	}

	sum := C.avx512_fuzzy_intersection_float64(
		(C.size_t)(size),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		intersectionPtr,
	)

	return float64(sum)
}

func (p *avx512Provider) SumFloat64(arr []float64) float64 {
	size := len(arr)

	sum := C.avx512_sum_float64(
		(C.size_t)(size),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}
