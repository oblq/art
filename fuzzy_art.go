package art

import (
	"fmt"
	"math"
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

	// Weight matrix - stores category prototypes
	W [][]float64
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
		batchSize:  16,
		wg:         sync.WaitGroup{},
		tPool: &sync.Pool{
			New: func() interface{} {
				return &activation{
					fi: make([]float64, inputLen*2),
				}
			},
		},
		rho:   rho,
		alpha: alpha,
		beta:  beta,
		W:     make([][]float64, 0),
	}, nil
}

// complementCode creates complement-coded representation of input vector.
// Complement coding is a common preprocessing step in ART models
// to prevent the "category proliferation problem."
// Complement coding achieve normalization while preserving amplitude information.
// Inputs preprocessed in complement coding are automatically normalized.
func (m *FuzzyART) complementCode(a []float64) []float64 {
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

// fuzzyIntersection populates the passed fuzzyIntersection arg with
// the element-wise min of the two input slices.
// Is the fuzzy `AND` operator.
// Measures the overlap between the input vector and the prototype vector.
// The fuzzy intersection slice is passed by reference to avoid unnecessary memory allocations.
func (m *FuzzyART) fuzzyIntersection(A, W, fuzzyIntersection []float64) {
	for i := range A {
		fuzzyIntersection[i] = math.Min(A[i], W[i])
	}
}

// l1Norm calculates the L1 norm (Manhattan distance) of a given slice of floats.
// Summing the components of the fuzzy intersection gives a measure of similarity
// that is analogous to an L1 norm in the context of complement-coded vectors.
func (m *FuzzyART) l1Norm(arr []float64) (norm float64) {
	for _, v := range arr {
		norm += v
	}

	return
}

// categoryChoice calculates the activation of a category based on the input vector.
// The fuzzyIntersection slice is passed by reference to avoid unnecessary memory allocations.
func (m *FuzzyART) categoryChoice(A, W, fuzzyIntersection []float64) (choice float64, fiNorm float64) {
	m.fuzzyIntersection(A, W, fuzzyIntersection)
	fiNorm = m.l1Norm(fuzzyIntersection)
	choice = fiNorm / (m.alpha + m.l1Norm(W))
	return
}

type activation struct {
	// fuzzy intersection
	fi []float64
	// L1 norm of the fuzzy intersection
	fiNorm float64
	// activation value, choice function value
	choice float64
	// index of the category weights
	j int
}

// activateCategories implements the recognition field functionality
// by computing activation values for each category based on the input vector.
// The sorting process also implicitly handles lateral inhibition by prioritizing
// the category with the highest activation, thereby inhibiting others.
func (m *FuzzyART) activateCategories(A []float64) (T []*activation) {
	T = make([]*activation, len(m.W))

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

	for jStart := 0; jStart < len(m.W); jStart += m.batchSize {
		jEnd := jStart + m.batchSize
		if jEnd > len(m.W) {
			jEnd = len(m.W)
		}

		m.wg.Add(1)
		// acquire a worker
		m.workerPool <- struct{}{}

		// spawn a goroutine to process a batch of categories
		go func(input []float64, categories [][]float64, startIndex int) {
			defer func() {
				// release the worker
				<-m.workerPool
			}()

			activations := make([]*activation, len(categories))
			for i, category := range categories {
				u := m.tPool.Get().(*activation)
				u.j = startIndex + i
				u.choice, u.fiNorm = m.categoryChoice(input, category, u.fi)
				activations[i] = u
			}
			activationsCh <- activations
		}(A, m.W[jStart:jEnd], jStart)
	}

	m.wg.Wait()

	// Sort category indices by activation values in descending order
	sort.SliceStable(T, func(a, b int) bool {
		// In case of equal activation values, sort by category index,
		// because older categories must have the priority.
		if T[a].choice == T[b].choice {
			return T[a].j < T[b].j
		}
		return T[a].choice > T[b].choice
	})

	return
}

// matchCriterion computes the matchCriterion function.
// The matchCriterion function calculates the resonance between the input vector and a category.
// The resonance is the ratio of the fuzzy intersection L1 norm to the input vector L1 norm.
func (m *FuzzyART) matchCriterion(fiNorm, aNorm float64) float64 {
	if fiNorm == 0 && aNorm == 0 {
		return 1
	} else {
		return fiNorm / aNorm
	}
}

// resonateOrReset implements the resonance or reset logic.
// If the best matching category passes the vigilance test (>= rho),
// its weights are updated to move closer to the input vector, facilitating learning.
// If it fails, the category is inhibited (temporarily ignored), and the next best category is tested,
// continuing until a suitable category is found or all are exhausted
// in which case a new category is created.
func (m *FuzzyART) resonateOrReset(
	A []float64,
	T []*activation,
) (categoryWeights []float64, categoryIndex int) {
	aNorm := m.l1Norm(A)

	for _, t := range T {
		resonance := m.matchCriterion(t.fiNorm, aNorm)
		if resonance >= m.rho {
			newW := make([]float64, len(A))
			for k := range newW {
				newW[k] = m.beta*t.fi[k] + (1-m.beta)*m.W[t.j][k]
			}

			m.W[t.j] = newW
			return newW, t.j
		}
	}

	// If no category meets the vigilance criterion, create a new category.
	// Fast commitment option, directly copy the input vector as the new category.
	m.W = append(m.W, A)

	return m.W[len(m.W)-1], len(m.W) - 1
}

// recover returns the activation instances to the pool for reuse.
func (m *FuzzyART) recover(T []*activation) {
	for _, t := range T {
		m.tPool.Put(t)
	}
}

// Fit implements the complete ART learning cycle.
func (m *FuzzyART) Fit(a []float64) ([]float64, int) {
	A := m.complementCode(a)
	T := m.activateCategories(A)
	defer m.recover(T)
	return m.resonateOrReset(A, T)
}

// Predict implements the recognition process with optional learning.
// It returns the weight vector of the best matching category and its index.
// If learn is true, it updates the weights of the matching category.
func (m *FuzzyART) Predict(a []float64, learn bool) ([]float64, int) {
	A := m.complementCode(a)
	T := m.activateCategories(A)
	defer m.recover(T)
	if !learn {
		return m.W[T[0].j], T[0].j
	}

	return m.resonateOrReset(A, T)
}

func (m *FuzzyART) Close() {
	close(m.workerPool)
}
