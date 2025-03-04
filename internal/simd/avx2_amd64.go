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
    static const size_t chunk_size = 8;  // Process 2 chunks (8 doubles) per iteration
    const size_t chunks = n / chunk_size;
    const size_t remainder_start = chunks * chunk_size;

    __m256d sum_vec0 = _mm256_setzero_pd();
    __m256d sum_vec1 = _mm256_setzero_pd();

    // Process 8 doubles at a time (2 chunks)
    for(size_t i = 0; i < chunks; ++i) {
        size_t offset = i * chunk_size;

        // First chunk
        __m256d a_vec0 = _mm256_loadu_pd(A + offset);
        __m256d w_vec0 = _mm256_loadu_pd(w + offset);
        __m256d min_vec0 = _mm256_min_pd(a_vec0, w_vec0);

        // Second chunk
        __m256d a_vec1 = _mm256_loadu_pd(A + offset + single_size);
        __m256d w_vec1 = _mm256_loadu_pd(w + offset + single_size);
        __m256d min_vec1 = _mm256_min_pd(a_vec1, w_vec1);

        // Store results in the output array if needed
        if (intersection_out != NULL) {
            _mm256_storeu_pd(intersection_out + offset, min_vec0);
            _mm256_storeu_pd(intersection_out + offset + single_size, min_vec1);
        }

        // Accumulate sums
        sum_vec0 = _mm256_add_pd(sum_vec0, min_vec0);
        sum_vec1 = _mm256_add_pd(sum_vec1, min_vec1);
    }

    // Combine the two sum vectors
    __m256d total_sum_vec = _mm256_add_pd(sum_vec0, sum_vec1);

    // Reduce sum vector to a single value
    double sum = 0.0;
    double *sum_arr = (double*)&total_sum_vec;
    for (int i = 0; i < 4; i++) {
        sum += sum_arr[i];
    }

    // Handle remaining elements
    for(size_t i = remainder_start; i < n; ++i) {
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
    static const size_t chunk_size = 8;  // Process 2 chunks (8 doubles) per iteration
    const size_t chunks = n / chunk_size;
    const size_t remainder_start = chunks * chunk_size;

    __m256d sum_vec0 = _mm256_setzero_pd();
    __m256d sum_vec1 = _mm256_setzero_pd();

    // Process 8 doubles at a time (2 chunks)
    for(size_t i = 0; i < chunks; ++i) {
        size_t offset = i * chunk_size;

        // Load and add first chunk
        __m256d arr_vec0 = _mm256_loadu_pd(arr + offset);
        sum_vec0 = _mm256_add_pd(sum_vec0, arr_vec0);

        // Load and add second chunk
        __m256d arr_vec1 = _mm256_loadu_pd(arr + offset + single_size);
        sum_vec1 = _mm256_add_pd(sum_vec1, arr_vec1);
    }

    // Combine the two sum vectors
    __m256d total_sum_vec = _mm256_add_pd(sum_vec0, sum_vec1);

    // Reduce sum vector to a single value
    double sum = 0.0;
    double *sum_arr = (double*)&total_sum_vec;
    for (int i = 0; i < 4; i++) {
        sum += sum_arr[i];
    }

    // Handle remaining elements
    for(size_t i = remainder_start; i < n; ++i) {
        sum += arr[i];
    }

    return sum;
}
*/
import "C"

// avx2 implements Provider with AVX2 instructions
type avx2 struct{}

// Override the factory function
func newAVX2Provider() Provider {
	return &avx2{}
}

func (p *avx2) FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
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

func (p *avx2) SumFloat64(arr []float64) float64 {
	size := len(arr)

	sum := C.avx2_sum_float64(
		(C.size_t)(size),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}
