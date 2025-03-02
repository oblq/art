package avx512

/*
#cgo CFLAGS: -mavx512f -mavx512vl -mavx512bw -mavx512vnni -mavx512dq
#cgo LDFLAGS: -lm
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>

int32_t avx2_dot_int8(const size_t n, int8_t *x, int8_t *y)
{
    static const size_t single_size = 32;
    const size_t end = n / single_size;
    const int16_t op4[16] = {[0 ... 15] = 1};
    __m256i *vx = (__m256i *)x;
    __m256i *vy = (__m256i *)y;
    __m256i vsum = {0};
    int32_t *t = (int32_t *)&vsum;
    for(size_t i = 0; i < end; ++i) {
        __m256i vresult1 = _mm256_maddubs_epi16(vx[i], vy[i]);
        __m256i vresult2 = _mm256_madd_epi16(vresult1, *(__m256i *)&op4);
        // trick here is to stop compiler over-optimize
        *(__m256i *)t = _mm256_add_epi32(vsum, vresult2);
    }
    int32_t sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += t[i];
    }
    return sum;
}

int32_t avx512_dot_vnni(const size_t n, int8_t *x, int8_t *y)
{
    static const size_t single_size = 64;
    const size_t end = n / single_size;
    __m512i *vx = (__m512i *)x;
    __m512i *vy = (__m512i *)y;
    __m512i vsum = {0};
    int32_t *t = (int32_t *)&vsum;
    for(size_t i = 0; i < end; ++i) {
        // trick here is to stop compiler over-optimize
        *(__m512i *)t = _mm512_dpbusds_epi32(vsum, vx[i], vy[i]);
    }
    int32_t sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += t[i];
    }
    return sum;
}

int32_t avx512_dot_int8(const size_t n, int8_t *x, int8_t *y)
{
    static const size_t single_size = 64;
    const size_t end = n / single_size;
    const int16_t op4[32] = {[0 ... 31] = 1};
    __m512i *vx = (__m512i *)x;
    __m512i *vy = (__m512i *)y;
    __m512i vsum = {0};
    int32_t *t = (int32_t *)&vsum;
    for(size_t i=0; i<end; ++i) {
        __m512i vresult1 = _mm512_maddubs_epi16(vx[i], vy[i]);
        __m512i vresult2 = _mm512_madd_epi16(vresult1, *(__m512i *)&op4);
        // trick here is to stop compiler over-optimize
        *(__m512i *)t = _mm512_add_epi32(vsum, vresult2);
    }
    int32_t sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += t[i];
    }
    return sum;
}

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
import (
	"math"
	"reflect"
	"unsafe"
)

func Malloc_int8(size int) []int8 {
	size_ := size
	size = align(size)
	ptr := C._mm_malloc((C.size_t)(C.sizeof_int8_t*size), 64)
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(ptr)),
		Len:  size,
		Cap:  size,
	}
	goSlice := *(*[]int8)(unsafe.Pointer(&hdr))
	if size_ != size {
		for i := size_; i < size; i++ {
			goSlice[i] = 0
		}
	}
	return goSlice
}

func Free_int8(v []int8) {
	C._mm_free(unsafe.Pointer(&v[0]))
}

func Make_int8(size int) []int8 {
	size_ := size
	size = align(size)
	goSlice := make([]int8, size)
	if size_ != size {
		for i := size_; i < size; i++ {
			goSlice[i] = 0
		}
	}
	return goSlice
}

func Dot_avx512_vnni(size int, x, y []int8) int32 {
	size = align(size)
	dot := C.avx512_dot_vnni((C.size_t)(size), (*C.int8_t)(&x[0]), (*C.int8_t)(&y[0]))
	return int32(dot)
}

func Dot_avx512_int8(size int, x, y []int8) int32 {
	size = align(size)
	dot := C.avx512_dot_int8((C.size_t)(size), (*C.int8_t)(&x[0]), (*C.int8_t)(&y[0]))
	return int32(dot)
}

func Dot_avx2_int8(size int, x, y []int8) int32 {
	size = align(size)
	dot := C.avx2_dot_int8((C.size_t)(size), (*C.int8_t)(&x[0]), (*C.int8_t)(&y[0]))
	return int32(dot)
}

// FuzzyIntersectionSum computes elementwise min between A and w and returns the sum
// If intersection_out is not nil, it also stores the intersection result
func FuzzyIntersectionSum(A, w []float64, intersection_out []float64) float64 {
	size := len(A)
	size = align64(size)

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

// SumFloat64 computes the sum of all elements in the array using AVX-512
func SumFloat64(arr []float64) float64 {
	size := len(arr)
	size = align64(size)

	sum := C.avx512_sum_float64(
		(C.size_t)(size),
		(*C.double)(&arr[0]),
	)

	return float64(sum)
}

func align(size int) int {
	return int(math.Ceil(float64(size)/64.0) * 64.0)
}

func align64(size int) int {
	return int(math.Ceil(float64(size)/8.0) * 8.0)
}
