//go:build amd64

package simd

import (
	"art/internal/simd/avx512"
	"golang.org/x/sys/cpu"
)

func hasAVX512() bool {
	return cpu.X86.HasAVX512 &&
		cpu.X86.HasAVX512F &&
		cpu.X86.HasAVX512DQ
}

func GetProvider() Provider {
	if hasAVX512() {
		return new(avx512.AVX512)
	}
	return nil
}
