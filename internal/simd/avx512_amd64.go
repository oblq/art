//go:build amd64

package simd

import (
	"math"
	"unsafe"

	"golang.org/x/sys/cpu"
)

/*
#cgo CFLAGS: -mavx512f -mavx512dq -mavx512vl -O3 -fPIC
#cgo CXXFLAGS: -mavx512f -mavx512dq -mavx512vl -O3 -fPIC -std=c++17
#cgo LDFLAGS: -lm -lstdc++
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>

// Computes the fuzzy intersection (elementwise min) between two arrays and returns the sum
double avx512_fuzzy_intersection_norm(const size_t n, double *A, double *w, double *fuzzy_intersection_out, double *w_norm_out)
{
    static const size_t single_size = 8; // 8 doubles per AVX-512 register
    static const size_t chunk_size = 2 * single_size; // Process 2 chunks (16 doubles) per iteration
    const size_t end = n / chunk_size;

    __m512d sum_vec1 = _mm512_setzero_pd();
    __m512d sum_vec2 = _mm512_setzero_pd();
    __m512d w_sum_vec1 = _mm512_setzero_pd();
    __m512d w_sum_vec2 = _mm512_setzero_pd();

    // Process 16 doubles (2 chunks) at a time
    for(size_t i = 0; i < end; ++i) {
        size_t offset = i * chunk_size;

        // First chunk
        __m512d a_vec1 = _mm512_loadu_pd(A + offset);
        __m512d w_vec1 = _mm512_loadu_pd(w + offset);
        __m512d min_vec1 = _mm512_min_pd(a_vec1, w_vec1);
        _mm512_storeu_pd(fuzzy_intersection_out + offset, min_vec1);
        sum_vec1 = _mm512_add_pd(sum_vec1, min_vec1);
        w_sum_vec1 = _mm512_add_pd(w_sum_vec1, w_vec1);

        // Second chunk
        __m512d a_vec2 = _mm512_loadu_pd(A + offset + single_size);
        __m512d w_vec2 = _mm512_loadu_pd(w + offset + single_size);
        __m512d min_vec2 = _mm512_min_pd(a_vec2, w_vec2);
        _mm512_storeu_pd(fuzzy_intersection_out + offset + single_size, min_vec2);
        sum_vec2 = _mm512_add_pd(sum_vec2, min_vec2);
        w_sum_vec2 = _mm512_add_pd(w_sum_vec2, w_vec2);
    }

    // Combine sums from both vectors
    double sum = _mm512_reduce_add_pd(sum_vec1) + _mm512_reduce_add_pd(sum_vec2);
    double w_sum = _mm512_reduce_add_pd(w_sum_vec1) + _mm512_reduce_add_pd(w_sum_vec2);

    // Handle remaining elements
    for(size_t i = end * chunk_size; i < n; ++i) {
        double min_val = A[i] < w[i] ? A[i] : w[i];
        fuzzy_intersection_out[i] = min_val;
        sum += min_val;
        w_sum += w[i];
    }

    // Store the sum of w elements to the output parameter
    *w_norm_out = w_sum;

    return sum;
}

// Computes the sum of an array using AVX-512 with 2 chunks per iteration
double avx512_sum(const size_t n, double *arr)
{
    static const size_t single_size = 8; // 8 doubles per AVX-512 register
    static const size_t chunk_count = 2; // Process 2 chunks per iteration
    const size_t chunk_size = single_size * chunk_count; // 16 doubles per iteration
    const size_t end = n / chunk_size;

    __m512d sum_vec1 = _mm512_setzero_pd();
    __m512d sum_vec2 = _mm512_setzero_pd();

    // Process 16 doubles at a time (2 chunks of 8 doubles each)
    for(size_t i = 0; i < end; ++i) {
        __m512d arr_vec1 = _mm512_loadu_pd(arr + i * chunk_size);
        __m512d arr_vec2 = _mm512_loadu_pd(arr + i * chunk_size + single_size);

        sum_vec1 = _mm512_add_pd(sum_vec1, arr_vec1);
        sum_vec2 = _mm512_add_pd(sum_vec2, arr_vec2);
    }

    // Combine the two sum vectors
    __m512d total_sum_vec = _mm512_add_pd(sum_vec1, sum_vec2);

    // Reduce sum vector to a single value using AVX-512 intrinsic
    double sum = _mm512_reduce_add_pd(total_sum_vec);

    // Handle remaining elements
    for(size_t i = end * chunk_size; i < n; ++i) {
        sum += arr[i];
    }

    return sum;
}

// update_fuzzy_weights updates weights using the formula:
// weights[i] = beta * fi[i] + (1-beta) * weights[i]
void update_fuzzy_weights(double* weights, const double* fi, double beta, int length) {
    int i = 0;

    // Get the value of (1-beta) once
    double oneminusbeta = 1.0 - beta;

    // Process 8 elements at a time using AVX512
    if (length >= 8) {
        __m512d beta_vec = _mm512_set1_pd(beta);
        __m512d oneminusbeta_vec = _mm512_set1_pd(oneminusbeta);

        for (; i <= length - 8; i += 8) {
            __m512d weights_vec = _mm512_loadu_pd(&weights[i]);
            __m512d fi_vec = _mm512_loadu_pd(&fi[i]);

            // beta * fi
            __m512d beta_fi = _mm512_mul_pd(beta_vec, fi_vec);

            // (1-beta) * weights
            __m512d oneminusbeta_weights = _mm512_mul_pd(oneminusbeta_vec, weights_vec);

            // beta * fi + (1-beta) * weights
            __m512d result = _mm512_add_pd(beta_fi, oneminusbeta_weights);

            // Store the result back to weights
            _mm512_storeu_pd(&weights[i], result);
        }
    }

    // Handle remaining elements
    for (; i < length; i++) {
        weights[i] = beta * fi[i] + oneminusbeta * weights[i];
    }
}

*/
import "C"

type AVX512 struct{}

func hasAVX512() bool {
	return cpu.X86.HasAVX512 &&
		cpu.X86.HasAVX512F &&
		cpu.X86.HasAVX512DQ
}

func GetProvider() Provider {
	if hasAVX512() {
		return new(AVX512)
	}
	return nil
}

// FuzzyIntersectionNorm computes elementwise min between A and w and returns the sum
// If intersection_out is not nil, it also stores the intersection result
func (p *AVX512) FuzzyIntersectionNorm(A, w []float64, fuzzyIntersectionOut []float64) (float64, float64) {
	size := len(A)

	// Ensure size is a multiple of 8 for AVX-512
	alignedSize := align64(size)

	var wNormOut C.double
	fiNormOut := C.avx512_fuzzy_intersection_norm(
		(C.size_t)(alignedSize),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		(*C.double)(&fuzzyIntersectionOut[0]),
		&wNormOut,
	)

	return float64(fiNormOut), float64(wNormOut)
}

// SumFloat64 computes the sum of all elements in the array using AVX-512
func (p *AVX512) SumFloat64(arr []float64) float64 {
	size := len(arr)
	alignedSize := align64(size)

	sum := C.avx512_sum(
		(C.size_t)(alignedSize),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}

// UpdateFuzzyWeights updates weights using AVX512 acceleration
// weights[i] = beta * fi[i] + (1-beta) * weights[i]
func (p *AVX512) UpdateFuzzyWeights(W []float64, fi []float64, beta float64) {
	size := len(W)
	alignedSize := align64(size)

	weightsPtr := (*C.double)(unsafe.Pointer(&W[0]))
	fiPtr := (*C.double)(unsafe.Pointer(&fi[0]))
	C.update_fuzzy_weights(weightsPtr, fiPtr, C.double(beta), C.int(alignedSize))
}

// align64 rounds up size to the nearest multiple of 8 (AVX-512 register can hold 8 doubles)
func align64(size int) int {
	return int(math.Ceil(float64(size)/8.0) * 8.0)
}
