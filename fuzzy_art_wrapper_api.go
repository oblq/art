package art

/*
#cgo CXXFLAGS: -std=c++17 -march=native -mavx512f
#cgo LDFLAGS: -lstdc++
#include "fuzzy_art_wrapper.h"
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

// FuzzyART is a Go wrapper for the C++ FuzzyART implementation
type FuzzyART struct {
	handle   C.FuzzyARTHandle
	inputLen int
}

// NewFuzzyART creates a new FuzzyART model
func NewFuzzyART(inputLen int, rho, alpha, beta float64) (*FuzzyART, error) {
	handle := C.FuzzyART_New(C.int(inputLen), C.float(rho), C.float(alpha), C.float(beta))
	if handle == nil {
		return nil, errors.New("failed to create FuzzyART model")
	}

	art := &FuzzyART{
		handle:   handle,
		inputLen: inputLen,
	}

	// Set up finalizer to free C++ memory when the Go object is garbage collected
	runtime.SetFinalizer(art, func(a *FuzzyART) {
		a.Close()
	})

	return art, nil
}

// Close frees the C++ FuzzyART model
func (a *FuzzyART) Close() {
	if a.handle != nil {
		C.FuzzyART_Free(a.handle)
		a.handle = nil
	}
}

// Fit trains the model with a single input vector
// Returns the weight vector of the matching category and the category index
func (a *FuzzyART) Fit(input []float64) ([]float64, int, error) {
	if a.handle == nil {
		return nil, -1, errors.New("FuzzyART model has been closed")
	}

	if len(input) != a.inputLen {
		return nil, -1, errors.New("input vector has incorrect length")
	}

	// Convert input to float32 array for C++
	inputFloat32 := make([]float32, len(input))
	for i, v := range input {
		inputFloat32[i] = float32(v)
	}

	// Prepare output weight vector (complement coded, so double length)
	weightsFloat32 := make([]float32, a.inputLen*2)

	// Call C function
	categoryIdx := int(C.FuzzyART_Fit(
		a.handle,
		(*C.float)(unsafe.Pointer(&inputFloat32[0])),
		C.int(len(inputFloat32)),
		(*C.float)(unsafe.Pointer(&weightsFloat32[0])),
		C.int(len(weightsFloat32)),
	))

	if categoryIdx == -1 {
		return nil, -1, errors.New("error during FuzzyART fit")
	}

	// Convert weights back to float64 for Go
	weights := make([]float64, len(weightsFloat32))
	for i, v := range weightsFloat32 {
		weights[i] = float64(v)
	}

	return weights, categoryIdx, nil
}

// Predict predicts the category for a single input vector
// If learn is true, the model will be updated
// Returns the weight vector of the matching category and the category index
func (a *FuzzyART) Predict(input []float64, learn bool) ([]float64, int, error) {
	if a.handle == nil {
		return nil, -1, errors.New("FuzzyART model has been closed")
	}

	if len(input) != a.inputLen {
		return nil, -1, errors.New("input vector has incorrect length")
	}

	// Convert input to float32 array for C++
	inputFloat32 := make([]float32, len(input))
	for i, v := range input {
		inputFloat32[i] = float32(v)
	}

	// Prepare output weight vector (complement coded, so double length)
	weightsFloat32 := make([]float32, a.inputLen*2)

	// Convert bool to int for C function
	learnInt := 0
	if learn {
		learnInt = 1
	}

	// Call C function
	categoryIdx := int(C.FuzzyART_Predict(
		a.handle,
		(*C.float)(unsafe.Pointer(&inputFloat32[0])),
		C.int(len(inputFloat32)),
		(*C.float)(unsafe.Pointer(&weightsFloat32[0])),
		C.int(len(weightsFloat32)),
		C.int(learnInt),
	))

	if categoryIdx == -1 {
		return nil, -1, errors.New("error during FuzzyART predict")
	}

	// Convert weights back to float64 for Go
	weights := make([]float64, len(weightsFloat32))
	for i, v := range weightsFloat32 {
		weights[i] = float64(v)
	}

	return weights, categoryIdx, nil
}

// GetNumCategories returns the number of categories in the model
func (a *FuzzyART) GetNumCategories() (int, error) {
	if a.handle == nil {
		return -1, errors.New("FuzzyART model has been closed")
	}

	return int(C.FuzzyART_GetNumCategories(a.handle)), nil
}

// GetInputDim returns the input dimension of the model
func (a *FuzzyART) GetInputDim() (int, error) {
	if a.handle == nil {
		return -1, errors.New("FuzzyART model has been closed")
	}

	return int(C.FuzzyART_GetInputDim(a.handle)), nil
}
