//go:build amd64

package simd

/*
#cgo CFLAGS: -mavx2
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

// Computes the fuzzy intersection (elementwise min) between two arrays and returns the sum
double avx2_fuzzy_intersection_float64(const size_t n, double *A, double *w, double *intersection_out)
{
    static const size_t single_size = 4; // 4 doubles per AVX2 register
    const size_t end = n / single_size;

    __m256d sum_vec = _mm256_setzero_pd();

    // Process 4 doubles at a time
    for(size_t i = 0; i < end; ++i) {
        __m256d a_vec = _mm256_loadu_pd(A + i * single_size);
        __m256d w_vec = _mm256_loadu_pd(w + i * single_size);

        // Compute min(A[i], w[j][i])
        __m256d min_vec = _mm256_min_pd(a_vec, w_vec);

        // Store result in the output array
        if (intersection_out != NULL) {
            _mm256_storeu_pd(intersection_out + i * single_size, min_vec);
        }

        // Accumulate sum
        sum_vec = _mm256_add_pd(sum_vec, min_vec);
    }

    // Reduce sum vector to a single value
    double sum = 0.0;
    double *sum_arr = (double*)&sum_vec;
    for (int i = 0; i < 4; i++) {
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

// Computes the sum of an array using AVX2
double avx2_sum_float64(const size_t n, double *arr)
{
    static const size_t single_size = 4; // 4 doubles per AVX2 register
    const size_t end = n / single_size;

    __m256d sum_vec = _mm256_setzero_pd();

    // Process 4 doubles at a time
    for(size_t i = 0; i < end; ++i) {
        __m256d arr_vec = _mm256_loadu_pd(arr + i * single_size);
        sum_vec = _mm256_add_pd(sum_vec, arr_vec);
    }

    // Reduce sum vector to a single value
    double sum = 0.0;
    double *sum_arr = (double*)&sum_vec;
    for (int i = 0; i < 4; i++) {
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

// avx2Provider implements Provider with AVX2 instructions
type avx2Provider struct{}

// Override the factory function
func newAVX2Provider() Provider {
	return &avx2Provider{}
}

func (p *avx2Provider) FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	size := len(A)

	var intersectionPtr *C.double
	if intersection_out != nil {
		intersectionPtr = (*C.double)(&intersection_out[0])
	}

	sum := C.avx2_fuzzy_intersection_float64(
		(C.size_t)(size),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		intersectionPtr,
	)

	return float64(sum)
}

func (p *avx2Provider) SumFloat64(arr []float64) float64 {
	size := len(arr)

	sum := C.avx2_sum_float64(
		(C.size_t)(size),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}
