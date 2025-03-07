//go:build darwin && arm64

package simd

func GetProvider() Provider {
	// todo: check if available
	return new(accelerate.Accelerate)
}
