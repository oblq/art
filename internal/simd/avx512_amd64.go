package simd

/*
#cgo CFLAGS: -mavx512f -mavx512vl -mavx512bw -mavx512vnni -mavx512dq
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

    // Process 8 doubles at a time with loop unrolling (2x)
    size_t i;
    for(i = 0; i + 1 < end; i += 2) {
        // Prefetch future data
        _mm_prefetch((const char*)(A + (i+4) * single_size), _MM_HINT_T0);
        _mm_prefetch((const char*)(w + (i+4) * single_size), _MM_HINT_T0);

        // First chunk
        __m512d a_vec1 = _mm512_loadu_pd(A + i * single_size);
        __m512d w_vec1 = _mm512_loadu_pd(w + i * single_size);
        __m512d min_vec1 = _mm512_min_pd(a_vec1, w_vec1);

        // Second chunk
        __m512d a_vec2 = _mm512_loadu_pd(A + (i+1) * single_size);
        __m512d w_vec2 = _mm512_loadu_pd(w + (i+1) * single_size);
        __m512d min_vec2 = _mm512_min_pd(a_vec2, w_vec2);

        // Store results if needed
        if (intersection_out != NULL) {
            _mm512_storeu_pd(intersection_out + i * single_size, min_vec1);
            _mm512_storeu_pd(intersection_out + (i+1) * single_size, min_vec2);
        }

        // Accumulate sums
        sum_vec = _mm512_add_pd(sum_vec, min_vec1);
        sum_vec = _mm512_add_pd(sum_vec, min_vec2);
    }

    // Handle remaining aligned chunk if any
    for(; i < end; ++i) {
        __m512d a_vec = _mm512_loadu_pd(A + i * single_size);
        __m512d w_vec = _mm512_loadu_pd(w + i * single_size);
        __m512d min_vec = _mm512_min_pd(a_vec, w_vec);

        if (intersection_out != NULL) {
            _mm512_storeu_pd(intersection_out + i * single_size, min_vec);
        }

        sum_vec = _mm512_add_pd(sum_vec, min_vec);
    }

    // Reduce sum vector to a single value using AVX-512 intrinsic
    double sum = _mm512_reduce_add_pd(sum_vec);

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

    // Process 8 doubles at a time with loop unrolling (2x)
    size_t i;
    for(i = 0; i + 1 < end; i += 2) {
        // Prefetch future data
        _mm_prefetch((const char*)(arr + (i+4) * single_size), _MM_HINT_T0);

        // Load and accumulate two chunks at once
        __m512d arr_vec1 = _mm512_loadu_pd(arr + i * single_size);
        __m512d arr_vec2 = _mm512_loadu_pd(arr + (i+1) * single_size);

        sum_vec = _mm512_add_pd(sum_vec, arr_vec1);
        sum_vec = _mm512_add_pd(sum_vec, arr_vec2);
    }

    // Handle remaining aligned chunk if any
    for(; i < end; ++i) {
        __m512d arr_vec = _mm512_loadu_pd(arr + i * single_size);
        sum_vec = _mm512_add_pd(sum_vec, arr_vec);
    }

    // Reduce sum vector to a single value using AVX-512 intrinsic
    double sum = _mm512_reduce_add_pd(sum_vec);

    // Handle remaining elements
    for(size_t i = end * single_size; i < n; ++i) {
        sum += arr[i];
    }

    return sum;
}
*/
import "C"
import (
	"math"
)

// avx2Provider implements Provider with AVX2 instructions
type avx512Provider struct{}

// Override the factory function
func newAVX512Provider() Provider {
	return new(avx512Provider)
}

// FuzzyIntersectionSum computes elementwise min between A and w and returns the sum
// If intersection_out is not nil, it also stores the intersection result
func (p *avx512Provider) FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	size := len(A)

	// Safety check
	if len(w) < size {
		size = len(w)
	}
	if intersection_out != nil && len(intersection_out) < size {
		intersection_out = nil // Don't write to too-small buffer
	}

	// Ensure size is a multiple of 8 for AVX-512
	alignedSize := align64(size)

	var intersectionPtr *C.double
	if intersection_out != nil {
		intersectionPtr = (*C.double)(&intersection_out[0])
	}

	sum := C.avx512_fuzzy_intersection_float64(
		(C.size_t)(alignedSize),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		intersectionPtr,
	)

	return float64(sum)
}

// SumFloat64 computes the sum of all elements in the array using AVX-512
func (p *avx512Provider) SumFloat64(arr []float64) float64 {
	size := len(arr)
	alignedSize := align64(size)

	sum := C.avx512_sum_float64(
		(C.size_t)(alignedSize),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}

// align64 rounds up size to the nearest multiple of 8 (AVX-512 register can hold 8 doubles)
func align64(size int) int {
	return int(math.Ceil(float64(size)/8.0) * 8.0)
}
