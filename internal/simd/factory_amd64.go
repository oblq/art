//go:build amd64

package simd

import (
	"golang.org/x/sys/cpu"
)

func hasAVX512() bool {
	return cpu.X86.HasAVX512 &&
		cpu.X86.HasAVX512F &&
		cpu.X86.HasAVX512VL &&
		cpu.X86.HasAVX512BW &&
		cpu.X86.HasAVX512VNNI &&
		cpu.X86.HasAVX512DQ
}

func hasAVX2() bool {
	return cpu.X86.HasAVX2
}

func GetProvider() Provider {
	if hasAVX512() {
		return new(avx512)
	} else if hasAVX2() {
		return new(avx2)
	}
	return nil
}
