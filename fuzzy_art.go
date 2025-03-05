package art

import (
	"fmt"
	"runtime"
	"sync"

	"art/internal/simd"
	"github.com/jfcg/sorty/v2"
)

type activation struct {
	// fuzzy intersection
	fi []float64
	// L1 norm of the fuzzy intersection
	fiNorm float64
	wNorm  float64
	// activation value, choice function value
	choice float64
	// index of the category weights
	j int
}

type job struct {
	input       []float64
	weights     [][]float64
	startIndex  int
	activations []*activation
}

// worker represents a worker goroutine
type worker struct {
	jobChan <-chan job
	wg      *sync.WaitGroup
}

// newWorker creates a new worker
func newWorker(jobChan <-chan job, wg *sync.WaitGroup) *worker {
	return &worker{
		jobChan: jobChan,
		wg:      wg,
	}
}

// start starts the worker goroutine
func (w *worker) start(art *FuzzyART) {
	go func() {
		for job := range w.jobChan {
			// Process the job
			for i, weights := range job.weights {
				u := job.activations[job.startIndex+i]
				u.j = job.startIndex + i
				u.fiNorm = simd.FuzzyIntersectionSum(job.input, weights, u.fi)
				u.wNorm = simd.SumFloat64(weights)
				u.choice = u.fiNorm / (art.alpha + u.wNorm)
			}
			w.wg.Done()
		}
	}()
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

	// M is the number of features of the input
	M int

	// W is the weight matrix - stores category prototypes
	W [][]float64
	T []*activation
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
		T:          make([]*activation, 0),
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

//// fuzzyIntersection populates the passed fuzzyIntersection arg with
//// the element-wise min of the two input slices.
//// Is the fuzzy `AND` operator.
//// Measures the overlap between the input vector and the prototype vector.
//// The fuzzy intersection slice is passed by reference to avoid unnecessary memory allocations.
//func (m *FuzzyART) fuzzyIntersection(A, W, fuzzyIntersection []float64) {
//	for i := range A {
//		fuzzyIntersection[i] = math.Min(A[i], W[i])
//	}
//}
//
//// l1Norm calculates the L1 norm (Manhattan distance) of a given slice of floats.
//// Summing the components of the fuzzy intersection gives a measure of similarity
//// that is analogous to an L1 norm in the context of complement-coded vectors.
//func (m *FuzzyART) l1Norm(arr []float64) (norm float64) {
//	for _, v := range arr {
//		norm += v
//	}
//
//	return
//}
//
//// categoryChoice calculates the activation of a category based on the input vector.
//// The fuzzyIntersection slice is passed by reference to avoid unnecessary memory allocations.
//func (m *FuzzyART) categoryChoice(A, W, fuzzyIntersection []float64) (choice, fiNorm, wNorm float64) {
//	m.fuzzyIntersection(A, W, fuzzyIntersection)
//	fiNorm = m.l1Norm(fuzzyIntersection)
//	wNorm = m.l1Norm(W)
//	choice = fiNorm / (m.alpha + wNorm)
//	return
//}

// activateCategories implements the recognition field functionality
// by computing activation values for each category based on the input vector.
// The sorting process also implicitly handles lateral inhibition by prioritizing
// the category with the highest activation, thereby inhibiting others.
func (m *FuzzyART) activateCategories(A []float64) {
	categoryChoice := func(input []float64, W [][]float64, startIndex int) {
		defer func() {
			// release the worker
			<-m.workerPool
			m.wg.Done()
		}()

		for i, w := range W {
			u := m.T[startIndex+i]
			u.j = startIndex + i
			//u.choice, u.fiNorm = m.categoryChoice(A, category, u.fi)
			u.fiNorm = simd.FuzzyIntersectionSum(A, w, u.fi)
			u.wNorm = simd.SumFloat64(w)
			u.choice = u.fiNorm / (m.alpha + u.wNorm)
		}
	}

	for jStart := 0; jStart < len(m.W); jStart += m.batchSize {
		jEnd := jStart + m.batchSize
		if jEnd > len(m.W) {
			jEnd = len(m.W)
		}

		m.wg.Add(1)
		// acquire a worker
		m.workerPool <- struct{}{}

		// spawn a goroutine to process a batch of categories
		go categoryChoice(A, m.W[jStart:jEnd], jStart)
	}

	m.wg.Wait()

	// Sort category indices by activation values in descending order
	//slices.SortFunc(m.T, func(a, b *activation) int {
	//	// In case of equal activation values, sort by category index,
	//	// because older categories must have the priority.
	//	if a.choice == b.choice {
	//		if a.j < b.j {
	//			return -1
	//		} else {
	//			return 1
	//		}
	//	}
	//	if a.choice > b.choice {
	//		return -1
	//	}
	//	return 1
	//})

	lsw := func(i, k, r, s int) bool {
		if m.T[i].choice == m.T[k].choice { // strict comparator like < or >
			if m.T[i].j < m.T[k].j { // strict comparator like < or >
				if r != s {
					m.T[r], m.T[s] = m.T[s], m.T[r]
				}
				return true
			}
			return false
		}
		if m.T[i].choice > m.T[k].choice { // strict comparator like < or >
			if r != s {
				m.T[r], m.T[s] = m.T[s], m.T[r]
			}
			return true
		}
		return false
	}
	sorty.Sort(len(m.T), lsw)
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

// extendClusterActivation initializes or expands
// the T slice to accommodate all weights in W.
func (m *FuzzyART) extendClusterActivation() {
	if len(m.T) < len(m.W) {
		// Keep existing activations but ensure T is at least as long as W
		if len(m.T) < len(m.W) {
			// Create a new slice with capacity for all weights
			newT := make([]*activation, len(m.W))
			// Copy existing activation instances
			copy(newT, m.T)
			// initialize the new activation
			lastIndex := len(m.T)
			newT[lastIndex] = &activation{
				fi: make([]float64, len(m.W[0])),
			}
			// Update T with the new slice
			m.T = newT
		}
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
) (resonance float64, categoryIndex int) {
	//aNorm := m.l1Norm(A)
	//aNorm := simd.SumFloat64(A)

	for _, t := range m.T {
		resonance = m.matchCriterion(t.fiNorm, float64(m.M))
		if resonance >= m.rho {
			newW := make([]float64, len(A))
			for k := range newW {
				newW[k] = m.beta*t.fi[k] + (1-m.beta)*m.W[t.j][k]
			}

			m.W[t.j] = newW
			return resonance, t.j
		}
	}

	// If no category meets the vigilance criterion, create a new category.
	// Fast commitment option, directly copy the input vector as the new category.
	m.W = append(m.W, A)
	m.extendClusterActivation()
	return resonance, len(m.W) - 1
}

// Fit implements the complete ART learning cycle.
func (m *FuzzyART) Fit(a []float64) (resonance float64, categoryIndex int) {
	A := m.complementCode(a)
	m.activateCategories(A)
	return m.resonateOrReset(A)
}

// Predict implements the recognition process with optional learning.
// It returns the weight vector of the best matching category and its index.
// If learn is true, it updates the weights of the matching category.
func (m *FuzzyART) Predict(a []float64, learn bool) (resonance float64, categoryIndex int) {
	A := m.complementCode(a)
	m.activateCategories(A)
	if !learn {
		//aNorm := simd.SumFloat64(A)
		resonance = m.matchCriterion(m.T[0].fiNorm, float64(m.M))
		return resonance, m.T[0].j
	}

	return m.resonateOrReset(A)
}

func (m *FuzzyART) Close() {
	close(m.workerPool)
}
