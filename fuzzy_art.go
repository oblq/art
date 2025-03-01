package art

import (
	"fmt"
	"runtime"
	"sort"
	"sync"
)

type FuzzyART struct {
	workerPool chan struct{}
	batchSize  int
	wg         sync.WaitGroup
	tPool      *sync.Pool

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
	// Purpose: Influences the activation function, affecting the competition among categories.
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

	// Flat array storage for all weights
	weights []float64
	// Number of categories
	numCategories int
	// Length of each weight vector
	weightVectorLength int
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

	// Calculate the length of complement-coded vectors
	weightVectorLength := inputLen * 2

	return &FuzzyART{
		workerPool: make(chan struct{}, runtime.NumCPU()),
		batchSize:  16,
		wg:         sync.WaitGroup{},
		tPool: &sync.Pool{
			New: func() interface{} {
				return &activation{
					fuzzyIntersection: make([]float64, weightVectorLength),
				}
			},
		},
		rho:                rho,
		alpha:              alpha,
		beta:               beta,
		weights:            make([]float64, 0),
		numCategories:      0,
		weightVectorLength: weightVectorLength,
	}, nil
}

// getWeightVector returns a slice view of the weights for category j
func (m *FuzzyART) getWeightVector(j int) []float64 {
	start := j * m.weightVectorLength
	end := start + m.weightVectorLength
	return m.weights[start:end]
}

// addCategory adds a new category with the given weights and returns its index
func (m *FuzzyART) addCategory(weights []float64) int {
	m.weights = append(m.weights, weights...)
	m.numCategories++
	return m.numCategories - 1
}

// updateWeightVector updates the weights for category j
func (m *FuzzyART) updateWeightVector(j int, newWeights []float64) {
	start := j * m.weightVectorLength
	copy(m.weights[start:start+m.weightVectorLength], newWeights)
}

// complementCode creates complement-coded representation of input vector.
func (m *FuzzyART) complementCode(a []float64) []float64 {
	// Create a new slice with double the length of the input slice
	A := make([]float64, m.weightVectorLength)
	for i, v := range a {
		// Copy the original value to the first half of the new slice
		A[i] = v
		// Calculate and store the complement (1 - X[i]) in the second half of the new slice
		A[i+len(a)] = 1 - v
	}
	return A
}

// choice compute the choice function.
func (m *FuzzyART) choice(A, W []float64, activation *activation) {
	activation.aNorm = 0
	activation.wNorm = 0
	activation.fuzzyIntersectionNorm = 0
	for i := range A {
		activation.aNorm += A[i]
		activation.wNorm += W[i]
		if A[i] < W[i] {
			activation.fuzzyIntersection[i] = A[i]
		} else {
			activation.fuzzyIntersection[i] = W[i]
		}
		activation.fuzzyIntersectionNorm += activation.fuzzyIntersection[i]
	}
	activation.t = activation.fuzzyIntersectionNorm / (m.alpha + activation.wNorm)
}

type activation struct {
	// 8-byte aligned fields
	t                     float64
	aNorm                 float64
	wNorm                 float64
	fuzzyIntersectionNorm float64
	// 8-byte aligned field
	fuzzyIntersection []float64
	// 4-byte field
	j int
	// 4-byte padding to ensure 8-byte alignment
	_ [4]byte
}

// categoryChoices implements the recognition field functionality
func (m *FuzzyART) categoryChoices(A []float64) (T []*activation) {
	T = make([]*activation, m.numCategories)

	// If no categories exist yet, return empty slice
	if m.numCategories == 0 {
		return T
	}

	activationsCh := make(chan []*activation, len(m.workerPool)*m.batchSize)
	defer close(activationsCh)

	go func() {
		for activations := range activationsCh {
			for _, a := range activations {
				T[a.j] = a
			}
			m.wg.Done()
		}
	}()

	for jStart := 0; jStart < m.numCategories; jStart += m.batchSize {
		jEnd := jStart + m.batchSize
		if jEnd > m.numCategories {
			jEnd = m.numCategories
		}

		m.wg.Add(1)
		// acquire a worker
		m.workerPool <- struct{}{}

		// spawn a goroutine to process a batch of categories
		go func(A []float64, startIndex, endIndex int) {
			defer func() {
				// release the worker
				<-m.workerPool
			}()

			activations := make([]*activation, endIndex-startIndex)
			for j := startIndex; j < endIndex; j++ {
				u := m.tPool.Get().(*activation)
				u.j = j

				// Get weight vector for this category
				W := m.getWeightVector(j)

				m.choice(A, W, u)
				activations[j-startIndex] = u
			}
			activationsCh <- activations
		}(A, jStart, jEnd)
	}

	m.wg.Wait()

	// Sort category indices by activation values in descending order
	sort.SliceStable(T, func(a, b int) bool {
		// In case of equal activation values, sort by category index,
		// because older categories must have the priority.
		if T[a].t == T[b].t {
			return T[a].j < T[b].j
		}
		return T[a].t > T[b].t
	})

	return
}

// match computes the match function.
func (m *FuzzyART) match(fiNorm, iNorm float64) float64 {
	if fiNorm == 0 && iNorm == 0 {
		return 1
	} else {
		return fiNorm / iNorm
	}
}

// resonateOrReset implements the resonance or reset logic.
func (m *FuzzyART) resonateOrReset(
	A []float64,
	T []*activation,
) ([]float64, int) {
	// Create a buffer for the new weights
	newWeights := make([]float64, m.weightVectorLength)

	for _, t := range T {
		resonance := m.match(t.fuzzyIntersectionNorm, t.aNorm)
		if resonance >= m.rho {
			// Get the current weights for this category
			W := m.getWeightVector(t.j)

			// Calculate new weights
			for k := range newWeights {
				newWeights[k] = m.beta*t.fuzzyIntersection[k] + (1-m.beta)*W[k]
			}

			// Update the weights for this category
			m.updateWeightVector(t.j, newWeights)

			// Return a view of the updated weights
			return m.getWeightVector(t.j), t.j
		}
	}

	// If no category meets the vigilance criterion, create a new category.
	categoryIndex := m.addCategory(A)
	return m.getWeightVector(categoryIndex), categoryIndex
}

// recover returns the activation instances to the pool for reuse.
func (m *FuzzyART) recover(T []*activation) {
	for _, t := range T {
		if t != nil {
			m.tPool.Put(t)
		}
	}
}

// Train implements the complete ART learning cycle.
func (m *FuzzyART) Train(a []float64) ([]float64, int) {
	A := m.complementCode(a)
	T := m.categoryChoices(A)
	defer m.recover(T)

	// If no categories exist yet, create the first one
	if m.numCategories == 0 {
		categoryIndex := m.addCategory(A)
		return m.getWeightVector(categoryIndex), categoryIndex
	}

	return m.resonateOrReset(A, T)
}

// Infer implements the recognition process with optional learning.
func (m *FuzzyART) Infer(a []float64, learn bool) ([]float64, int) {
	A := m.complementCode(a)
	T := m.categoryChoices(A)
	defer m.recover(T)

	// If no categories exist yet, create the first one if learning is enabled
	if m.numCategories == 0 {
		if learn {
			categoryIndex := m.addCategory(A)
			return m.getWeightVector(categoryIndex), categoryIndex
		}
		// Return empty result if no categories and not learning
		return make([]float64, m.weightVectorLength), -1
	}

	if !learn {
		return m.getWeightVector(T[0].j), T[0].j
	}

	return m.resonateOrReset(A, T)
}

func (m *FuzzyART) Close() {
	close(m.workerPool)
}
