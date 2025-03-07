package art

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"art/internal/simd"
)

type fuzzyActivation struct {
	// fuzzy intersection
	fi []float64
	// L1 norm of the fuzzy intersection
	fiNorm float64
	wNorm  float64
	// activation value, choice function value
	activation float64
	// index of the category weights
	j int
}

type FuzzyART struct {
	workerPool chan struct{}
	batchSize  int
	wg         sync.WaitGroup

	// Vigilance parameter - controls category granularity
	// Recommended value: 0.8
	// Range: 0.0 to 1.0
	// Typical Values: 0.5 to 0.9
	// Purpose: Determines the strictness of category matching. Higher values mean stricter matching criteria.
	// Adjustment:
	// Increase rho to make the model more selective, creating more categories.
	// Decrease rho to allow more generalization, creating fewer categories.
	rho float64

	// Choice parameter - influences category competition
	// Recommended value: 0.01
	// Range: > 0.0
	// Typical Values: 0.0001 to 0.1
	// Purpose: Influences the fuzzyActivation function, affecting the competition among categories.
	// Adjustment:
	// Higher alpha values generates lower categories competition (creates fewer).
	// Lower alpha values (closer to 0) higher categories competition (creates more).
	alpha float64

	// Learning rate - controls weight update speed
	// When beta == 1 is called fast-learning, it becomes sensitive to noise.
	// Lower β values provide more gradual, stable learning but require more training iterations.
	// Recommended value: 1.0
	// Range: 0.0 to 1.0
	// Purpose: Controls the rate at which the weights are updated during training.
	// Adjustment:
	// Increase beta for faster learning, which can be useful for rapidly changing environments.
	// Decrease beta for more stable learning, which can be beneficial for more stable environments.
	beta float64

	// M is the number of features of the input
	M int

	// W is the weight matrix - stores category prototypes
	W [][]float64
	T []*fuzzyActivation
}

func NewFuzzyART(inputLen int, rho float64, alpha float64, beta float64) (*FuzzyART, error) {
	if rho < 0 || rho > 1 {
		return nil, fmt.Errorf("vigilance parameter (rho) must be between 0 and 1, got %f", rho)
	}
	if alpha <= 0 {
		return nil, fmt.Errorf("choice parameter (alpha) must be positive, got %f", alpha)
	}
	if beta <= 0 || beta > 1 {
		return nil, fmt.Errorf("learning rate (beta) must be between 0 and 1, got %f", beta)
	}

	return &FuzzyART{
		workerPool: make(chan struct{}, runtime.NumCPU()),
		batchSize:  64,
		wg:         sync.WaitGroup{},
		rho:        rho,
		alpha:      alpha,
		beta:       beta,
		M:          inputLen,
		W:          make([][]float64, 0),
		T:          make([]*fuzzyActivation, 0),
	}, nil
}

// complementCode creates complement-coded representation of input vector.
// Complement coding is a common preprocessing step in ART models
// to prevent the "category proliferation problem."
// Complement coding achieve normalization while preserving amplitude information.
// Inputs preprocessed in complement coding are automatically normalized.
func (f *FuzzyART) complementCode(a []float64) []float64 {
	// Create a new slice with double the length of the input slice
	A := make([]float64, len(a)*2)
	for i, v := range a {
		// Copy the original value to the first half of the new slice
		A[i] = v
		// Calculate and store the complement (1 - X[i]) in the second half of the new slice
		A[i+len(a)] = 1 - v
	}

	return A
}

// activateCategories implements the recognition field functionality
// by computing activation values for each category based on the input vector.
// The sorting process also implicitly handles lateral inhibition by prioritizing
// the category with the highest activation, thereby inhibiting others.
func (f *FuzzyART) activateCategories(A []float64) {
	categoryChoice := func(input []float64, W [][]float64, startIndex int) {
		defer func() {
			// release the worker
			<-f.workerPool
			f.wg.Done()
		}()

		for i, w := range W {
			u := f.T[startIndex+i]
			u.j = startIndex + i
			u.fiNorm, u.wNorm = simd.Shared.FuzzyIntersectionNorm(A, w, u.fi)
			u.activation = u.fiNorm / (f.alpha + u.wNorm)
		}
	}

	for jStart := 0; jStart < len(f.W); jStart += f.batchSize {
		jEnd := jStart + f.batchSize
		if jEnd > len(f.W) {
			jEnd = len(f.W)
		}

		f.wg.Add(1)
		// acquire a worker
		f.workerPool <- struct{}{}

		// spawn a goroutine to process a batch of categories
		go categoryChoice(A, f.W[jStart:jEnd], jStart)
	}

	f.wg.Wait()
	f.sortCategoriesByActivation()
}

func (f *FuzzyART) sortCategoriesByActivation() {
	// Sort category val values indices by val values in descending order
	slices.SortFunc(f.T, func(a, b *fuzzyActivation) int {
		// In case of equal activation values, sort by category index,
		// because older categories must have the priority.
		if a.activation == b.activation {
			if a.j < b.j {
				return -1
			} else {
				return 1
			}
		}
		if a.activation > b.activation {
			return -1
		}
		return 1
	})
}

// resonance calculates the resonance between the input vector and a category.
// The resonance is the ratio of the fuzzy intersection L1 norm to the input vector L1 norm.
func (f *FuzzyART) resonance(fiNorm, aNorm float64) float64 {
	if fiNorm == 0 && aNorm == 0 {
		return 1
	}

	return fiNorm / aNorm
}

// extendClusterActivation initializes or expands
// the T slice to accommodate all weights in W.
func (f *FuzzyART) extendClusterActivation() {
	if len(f.T) < len(f.W) {
		// Keep existing activations but ensure T is at least as long as W
		if len(f.T) < len(f.W) {
			// Create a new slice with capacity for all weights
			newT := make([]*fuzzyActivation, len(f.W))
			// Copy existing fuzzyActivation instances
			copy(newT, f.T)
			// initialize the new fuzzyActivation
			lastIndex := len(f.T)
			newT[lastIndex] = &fuzzyActivation{
				fi: make([]float64, len(f.W[0])),
			}
			// Update T with the new slice
			f.T = newT
		}
	}
}

func (f *FuzzyART) appendNewCategory(A []float64) int {
	f.W = append(f.W, A)
	f.extendClusterActivation()
	return len(f.W) - 1
}

// resonateOrReset implements the resonance or reset logic.
// If the best matching category passes the vigilance test (>= rho),
// its weights are updated to move closer to the input vector, facilitating learning.
// If it fails, the category is inhibited (temporarily ignored), and the next best category is tested,
// continuing until a suitable category is found or all are exhausted
// in which case a new category is created.
func (f *FuzzyART) resonateOrReset(A []float64) (maxResonance float64, categoryIndex int) {
	aNorm := simd.Shared.SumFloat64(A)

	for _, t := range f.T {
		resonance := f.resonance(t.fiNorm, aNorm)
		maxResonance = math.Max(maxResonance, resonance)
		if resonance >= f.rho {
			simd.Shared.UpdateFuzzyWeights(f.W[t.j], t.fi, f.beta)
			return resonance, t.j
		}
	}

	// If no category meets the vigilance criterion, create a new category.
	// Fast commitment option, directly copy the input vector as the new category.
	categoryIndex = f.appendNewCategory(A)
	return
}

// Fit implements the complete ART learning cycle.
func (f *FuzzyART) Fit(a []float64) (resonance float64, categoryIndex int) {
	A := f.complementCode(a)
	f.activateCategories(A)
	return f.resonateOrReset(A)
}

// Predict implements the recognition process with optional learning.
// It returns the weight vector of the best matching category and its index.
// If learn is true, it updates the weights of the matching category.
func (f *FuzzyART) Predict(a []float64, learn bool) (resonance float64, categoryIndex int) {
	A := f.complementCode(a)
	f.activateCategories(A)
	if !learn {
		aNorm := simd.Shared.SumFloat64(A)
		activation := f.T[0]
		resonance = f.resonance(activation.fiNorm, aNorm)
		return resonance, activation.j
	}

	return f.resonateOrReset(A)
}

func (f *FuzzyART) Close() {
	close(f.workerPool)
}
