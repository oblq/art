//go:build amd64

package simd

/*
#cgo CFLAGS: -mavx512f -mavx512dq
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>

// Computes the fuzzy intersection (elementwise min) between two arrays and returns the sum
double avx512_fuzzy_intersection_float64(const size_t n, double *A, double *w, double *intersection_out)
{
    static const size_t single_size = 8; // 8 doubles per AVX-512 register
    static const size_t chunk_size = 2 * single_size; // Process 2 chunks (16 doubles) per iteration
    const size_t end = n / chunk_size;

    __m512d sum_vec1 = _mm512_setzero_pd();
    __m512d sum_vec2 = _mm512_setzero_pd();

    // Process 16 doubles (2 chunks) at a time
    for(size_t i = 0; i < end; ++i) {
        size_t offset = i * chunk_size;

        // First chunk
        __m512d a_vec1 = _mm512_loadu_pd(A + offset);
        __m512d w_vec1 = _mm512_loadu_pd(w + offset);
        __m512d min_vec1 = _mm512_min_pd(a_vec1, w_vec1);
        _mm512_storeu_pd(intersection_out + offset, min_vec1);
        sum_vec1 = _mm512_add_pd(sum_vec1, min_vec1);

        // Second chunk
        __m512d a_vec2 = _mm512_loadu_pd(A + offset + single_size);
        __m512d w_vec2 = _mm512_loadu_pd(w + offset + single_size);
        __m512d min_vec2 = _mm512_min_pd(a_vec2, w_vec2);
        _mm512_storeu_pd(intersection_out + offset + single_size, min_vec2);
        sum_vec2 = _mm512_add_pd(sum_vec2, min_vec2);
    }

    // Combine sums from both vectors
    double sum = _mm512_reduce_add_pd(sum_vec1) + _mm512_reduce_add_pd(sum_vec2);

    // Handle remaining elements
    for(size_t i = end * chunk_size; i < n; ++i) {
        double min_val = A[i] < w[i] ? A[i] : w[i];
        intersection_out[i] = min_val;
        sum += min_val;
    }

    return sum;
}

// Computes the sum of an array using AVX-512 with 2 chunks per iteration
double avx512_sum_float64(const size_t n, double *arr)
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

// Find top k activations using a stream-based approach
void avx512_top_k_activations(
    const size_t n,
    double *choices,
    int *indices,
    double *top_k_values,
    int *top_k_indices,
    const size_t k)
{
    // Initialize with invalid values
    for (size_t i = 0; i < k; i++) {
        top_k_values[i] = -INFINITY;
        top_k_indices[i] = -1;
    }

    // Stream through all activations, maintaining top k
    for (size_t i = 0; i < n; i++) {
        double val = choices[i];
        int idx = indices[i];

        // Find insertion position based on value and tie-breaking
        size_t pos = k;
        for (size_t j = 0; j < k; j++) {
            if (val > top_k_values[j] ||
                (val == top_k_values[j] && idx < top_k_indices[j])) {
                pos = j;
                break;
            }
        }

        // If an insertion position was found, shift and insert
        if (pos < k) {
            // Shift elements down
            for (size_t j = k-1; j > pos; j--) {
                top_k_values[j] = top_k_values[j-1];
                top_k_indices[j] = top_k_indices[j-1];
            }
            // Insert the new value
            top_k_values[pos] = val;
            top_k_indices[pos] = idx;
        }
    }
}
*/
import "C"
import (
	"math"
	"unsafe"

	"golang.org/x/sys/cpu"
)

// avx2 implements Provider with AVX2 instructions
type avx512 struct{}

// Override the factory function
func newAVX512Provider() Provider {
	return new(avx512)
}

// FuzzyIntersectionSum computes elementwise min between A and w and returns the sum
// If intersection_out is not nil, it also stores the intersection result
func (p *avx512) FuzzyIntersectionSum(A, w []float64, intersectionOut []float64) float64 {
	size := len(A)

	// Ensure size is a multiple of 8 for AVX-512
	alignedSize := align64(size)

	sum := C.avx512_fuzzy_intersection_float64(
		(C.size_t)(alignedSize),
		(*C.double)(&A[0]),
		(*C.double)(&w[0]),
		(*C.double)(&intersectionOut[0]),
	)

	return float64(sum)
}

// SumFloat64 computes the sum of all elements in the array using AVX-512
func (p *avx512) SumFloat64(arr []float64) float64 {
	size := len(arr)
	alignedSize := align64(size)

	sum := C.avx512_sum_float64(
		(C.size_t)(alignedSize),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}

// TopKActivations finds the top k activations and their indices
// Returns two slices: values and their corresponding indices
func (p *avx512) TopKActivations(choices []float64, indices []int, k int) ([]float64, []int) {
	n := len(choices)
	if n == 0 || k <= 0 {
		return []float64{}, []int{}
	}

	// Ensure k doesn't exceed array length
	if k > n {
		k = n
	}

	// Prepare output arrays
	topValues := make([]float64, k)
	topIndices := make([]int, k)

	// Convert Go slices to C arrays
	choicesPtr := (*C.double)(unsafe.Pointer(&choices[0]))
	indicesPtr := (*C.int)(unsafe.Pointer(&indices[0]))
	topValuesPtr := (*C.double)(unsafe.Pointer(&topValues[0]))
	topIndicesPtr := (*C.int)(unsafe.Pointer(&topIndices[0]))

	// Call C function
	C.avx512_top_k_activations(
		C.size_t(n),
		choicesPtr,
		indicesPtr,
		topValuesPtr,
		topIndicesPtr,
		C.size_t(k),
	)

	return topValues, topIndices
}

// align64 rounds up size to the nearest multiple of 8 (AVX-512 register can hold 8 doubles)
func align64(size int) int {
	return int(math.Ceil(float64(size)/8.0) * 8.0)
}

func hasAVX512() bool {
	return cpu.X86.HasAVX512 &&
		cpu.X86.HasAVX512F &&
		cpu.X86.HasAVX512DQ
}

func GetProvider() Provider {
	if hasAVX512() {
		return new(avx512)
	}
	return nil
}
