package simd

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

func TestFuzzyIntersectionNorm(t *testing.T) {
	for _, size := range []int{7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 256} {
		t.Run("size="+strconv.Itoa(size), func(t *testing.T) {
			a := make([]float64, size)
			w := make([]float64, size)
			intersection := make([]float64, size)

			var expectedSum float64
			for i := 0; i < size; i++ {
				a[i] = rand.Float64() * 10
				w[i] = rand.Float64() * 10

				// Expected intersection and sum
				minimum := math.Min(a[i], w[i])
				expectedSum += minimum
				intersection[i] = 0 // Initialize to 0 for proper comparison later
			}

			// Call our optimized function
			fiNorm, _ := Shared.FuzzyIntersectionNorm(a, w, intersection)

			// Verify sum
			if math.Abs(expectedSum-fiNorm) > 1e-10 {
				t.Errorf("FuzzyIntersectionSum should return sum %.10f, but got %.10f",
					expectedSum, fiNorm)
			}

			// Verify intersection values
			for i := 0; i < size; i++ {
				expected := math.Min(a[i], w[i])
				if math.Abs(expected-intersection[i]) > 1e-10 {
					t.Errorf("FuzzyIntersection at index %d should be %.10f, but got %.10f",
						i, expected, intersection[i])
				}
			}
		})
	}
}

func TestSumFloat64(t *testing.T) {
	for _, size := range []int{7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 256} {
		t.Run("size="+strconv.Itoa(size), func(t *testing.T) {
			arr := make([]float64, size)

			var expectedSum float64
			for i := 0; i < size; i++ {
				arr[i] = rand.Float64() * 10
				expectedSum += arr[i]
			}

			// Call our optimized function
			resultSum := Shared.SumFloat64(arr)

			// Verify sum
			if math.Abs(expectedSum-resultSum) > 1e-10 {
				t.Errorf("SumFloat64 should return %.10f, but got %.10f", expectedSum, resultSum)
			}
		})
	}
}

func BenchmarkFuzzyIntersectionSum(b *testing.B) {
	benchSizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range benchSizes {
		b.Run("size="+strconv.Itoa(size), func(b *testing.B) {
			a := make([]float64, size)
			w := make([]float64, size)
			intersection := make([]float64, size)

			for i := 0; i < size; i++ {
				a[i] = rand.Float64() * 10
				w[i] = rand.Float64() * 10
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Shared.FuzzyIntersectionNorm(a, w, intersection)
			}
		})
	}
}

func BenchmarkSumFloat64(b *testing.B) {
	benchSizes := []int{8, 64, 256, 1024, 4096}

	for _, size := range benchSizes {
		b.Run("size="+strconv.Itoa(size), func(b *testing.B) {
			arr := make([]float64, size)

			for i := 0; i < size; i++ {
				arr[i] = rand.Float64() * 10
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Shared.SumFloat64(arr)
			}
		})
	}
}
