package avx512

import (
	"math"
	"math/rand"
	"testing"
)

const benchsize = 1036

func TestDotAvx512Int8(t *testing.T) {
	for _, size := range []int{63, 64, 127, 128, 256} {
		func(size int) {
			vx := Make_int8(size)
			vy := Make_int8(size)

			var truth int32
			for i := 0; i < size; i++ {
				vx[i] = int8(rand.Intn(127))
				vy[i] = int8(rand.Intn(127))
				truth += int32(vx[i]) * int32(vy[i])
			}

			result := Dot_avx512_int8(size, vx, vy)
			if truth != result {
				t.Errorf("Dot should return %d, but %d", truth, result)
			}
		}(size)
	}
}

func TestDotAvx512Vnni(t *testing.T) {
	for _, size := range []int{63, 64, 127, 128, 256} {
		func(size int) {
			vx := Make_int8(size)
			vy := Make_int8(size)

			var truth int32
			for i := 0; i < size; i++ {
				vx[i] = int8(rand.Intn(127))
				vy[i] = int8(rand.Intn(127))
				truth += int32(vx[i]) * int32(vy[i])
			}

			result := Dot_avx512_vnni(size, vx, vy)
			if truth != result {
				t.Errorf("Dot should return %d, but %d", truth, result)
			}
		}(size)
	}
}

func TestDotAvx2Int8(t *testing.T) {
	for _, size := range []int{63, 64, 127, 128, 256} {
		func(size int) {
			vx := Make_int8(size)
			vy := Make_int8(size)

			var truth int32
			for i := 0; i < size; i++ {
				vx[i] = int8(rand.Intn(127))
				vy[i] = int8(rand.Intn(127))
				truth += int32(vx[i]) * int32(vy[i])
			}

			result := Dot_avx2_int8(size, vx, vy)
			if truth != result {
				t.Errorf("Dot should return %d, but %d", truth, result)
			}
		}(size)
	}
}

func BenchmarkAvx2DotInt8(b *testing.B) {
	size := benchsize
	vx := Make_int8(size)
	vy := Make_int8(size)
	for i := 0; i < size; i++ {
		vx[i] = int8(rand.Intn(127))
		vy[i] = int8(rand.Intn(127))
	}
	b.SetBytes(int64(size))
	b.ResetTimer()
	var result int32 = 0
	for i := 0; i < b.N; i++ {
		result += Dot_avx2_int8(size, vx, vy)
		vx[i%size] = int8(result)
		vy[i%size] = int8(result)
	}
}

func BenchmarkAVX512DotInt8(b *testing.B) {
	size := benchsize
	vx := Make_int8(size)
	vy := Make_int8(size)
	for i := 0; i < size; i++ {
		vx[i] = int8(rand.Intn(127))
		vy[i] = int8(rand.Intn(127))
	}
	b.SetBytes(int64(size))
	b.ResetTimer()
	var result int32 = 0
	for i := 0; i < b.N; i++ {
		result += Dot_avx512_int8(size, vx, vy)
		vx[i%size] = int8(result)
		vy[i%size] = int8(result)
	}
}

func BenchmarkAVX512DotVnni(b *testing.B) {
	size := benchsize
	vx := Make_int8(size)
	vy := Make_int8(size)
	for i := 0; i < size; i++ {
		vx[i] = int8(rand.Intn(127))
		vy[i] = int8(rand.Intn(127))
	}
	b.SetBytes(int64(size))
	b.ResetTimer()
	var result int32 = 0
	for i := 0; i < b.N; i++ {
		result += Dot_avx512_vnni(size, vx, vy)
		vx[i%size] = int8(result)
		vy[i%size] = int8(result)
	}
}

// New benchmark for FuzzyIntersectionSum
func BenchmarkFuzzyIntersectionSum(b *testing.B) {
	size := benchsize
	a := make([]float64, size)
	w := make([]float64, size)
	intersection := make([]float64, size)

	for i := 0; i < size; i++ {
		a[i] = rand.Float64() * 10
		w[i] = rand.Float64() * 10
	}

	b.SetBytes(int64(size * 8)) // 8 bytes per float64
	b.ResetTimer()

	var result float64
	for i := 0; i < b.N; i++ {
		result += FuzzyIntersectionSum(a, w, intersection)
		a[i%size] = result * 0.0001 // small modification to avoid compiler optimization
		w[i%size] = result * 0.0001
	}
}

// New benchmark for SumFloat64
func BenchmarkSumFloat64(b *testing.B) {
	size := benchsize
	arr := make([]float64, size)

	for i := 0; i < size; i++ {
		arr[i] = rand.Float64() * 10
	}

	b.SetBytes(int64(size * 8)) // 8 bytes per float64
	b.ResetTimer()

	var result float64
	for i := 0; i < b.N; i++ {
		result += SumFloat64(arr)
		arr[i%size] = result * 0.0001 // small modification to avoid compiler optimization
	}
}

// Benchmark the naive implementation for comparison
func BenchmarkNaiveFuzzyIntersection(b *testing.B) {
	size := benchsize
	a := make([]float64, size)
	w := make([]float64, size)
	intersection := make([]float64, size)

	for i := 0; i < size; i++ {
		a[i] = rand.Float64() * 10
		w[i] = rand.Float64() * 10
	}

	b.SetBytes(int64(size * 8)) // 8 bytes per float64
	b.ResetTimer()

	var result float64
	for i := 0; i < b.N; i++ {
		// Naive implementation
		sum := 0.0
		for j := 0; j < size; j++ {
			min := math.Min(a[j], w[j])
			intersection[j] = min
			sum += min
		}
		result += sum
		a[i%size] = result * 0.0001 // small modification to avoid compiler optimization
		w[i%size] = result * 0.0001
	}
}

// Benchmark the naive implementation of sum for comparison
func BenchmarkNaiveSum(b *testing.B) {
	size := benchsize
	arr := make([]float64, size)

	for i := 0; i < size; i++ {
		arr[i] = rand.Float64() * 10
	}

	b.SetBytes(int64(size * 8)) // 8 bytes per float64
	b.ResetTimer()

	var result float64
	for i := 0; i < b.N; i++ {
		// Naive implementation
		sum := 0.0
		for j := 0; j < size; j++ {
			sum += arr[j]
		}
		result += sum
		arr[i%size] = result * 0.0001 // small modification to avoid compiler optimization
	}
}
