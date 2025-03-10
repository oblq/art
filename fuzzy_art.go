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
	// L1 norm of the relative category weights
	wNorm float64
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
	// Recommended value: 0.86
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
	// Typical Values: 0.0001 to 10
	// Purpose: Influences the activation function, affecting the competition among categories.
	// Adjustment:
	// Higher alpha values makes the choice more dependent on the match between input and weight
	// instead of considering the size of the input vector,
	// tend to favor the selection of existing categories over creating new ones.
	// Lower alpha values bias toward selecting categories with larger weight vectors.
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

	// T is the activation list - stores category activations
	// This is an internal variable used for internal calculations
	// and should not be manipulated externally.
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
	A := make([]float64, len(a)*2)
	for i, v := range a {
		A[i] = v
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

func (f *FuzzyART) appendNewCategory(A []float64) int {
	f.W = append(f.W, A)
	f.T = append(f.T, &fuzzyActivation{
		fi: make([]float64, len(f.W[0])),
	})
	return len(f.W) - 1
}

// resonateOrReset implements the resonance or reset logic.
// If the best matching category passes the vigilance test (>= rho),
// its weights are updated to move closer to the input vector, facilitating learning.
// If it fails, the category is inhibited (temporarily ignored),
// and the next best category is tested,
// continuing until a suitable category is found or all are exhausted
// in which case a new category is created.
func (f *FuzzyART) resonateOrReset(A []float64) (maxResonance float64, categoryIndex int) {
	aNorm := simd.Shared.SumFloat64(A)

	for _, t := range f.T {
		resonance := f.resonance(t.fiNorm, aNorm)
		if resonance >= f.rho {
			simd.Shared.UpdateFuzzyWeights(f.W[t.j], t.fi, f.beta)
			return resonance, t.j
		}
		maxResonance = math.Max(maxResonance, resonance)
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
// It returns the resonance value and the index of the best matching category.
// If learn is true, it also updates the weights of the matching category.
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
