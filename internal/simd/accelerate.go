//go:build darwin && arm64

package simd

import "art/internal/simd/accelerate"

func GetProvider() Provider {
	// todo: check if available
	return new(accelerate.Accelerate)
}
