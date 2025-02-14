package art

import (
	"fmt"
	"runtime"
	"sort"
	"sync"
)

type FuzzyART struct {
	workerPoolSize         int
	batchSize              int
	fuzzyIntersectionsPool *sync.Pool
	workerPool             chan struct{}
	wg                     sync.WaitGroup

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

func New(inputLen int, rho float64, alpha float64, beta float64) (*FuzzyART, error) {
	if rho < 0 || rho > 1 {
		return nil, fmt.Errorf("vigilance parameter (rho) must be between 0 and 1, got %f", rho)
	}
	if alpha <= 0 {
		return nil, fmt.Errorf("choice parameter (alpha) must be positive, got %f", alpha)
	}
	if beta <= 0 || beta > 1 {
		return nil, fmt.Errorf("learning rate (beta) must be between 0 and 1, got %f", beta)
	}

	workerPoolSize := runtime.NumCPU()
	if workerPoolSize >= 4 {
		workerPoolSize -= 2
	}
	batchSize := 8

	fuzzyIntersectionsPool := &sync.Pool{
		New: func() interface{} {
			return make([]float64, inputLen*2)
		},
	}

	return &FuzzyART{
		workerPoolSize:         workerPoolSize,
		batchSize:              batchSize,
		fuzzyIntersectionsPool: fuzzyIntersectionsPool,
		workerPool:             make(chan struct{}, workerPoolSize),
		wg:                     sync.WaitGroup{},
		rho:                    rho,
		alpha:                  alpha,
		beta:                   beta,
		W:                      make([][]float64, 0),
	}, nil
}

// complementCode creates complement-coded representation of input vector.
// Complement coding is a common preprocessing step in ART models
// to prevent the "category proliferation problem."
// Complement coding achieve normalization while preserving amplitude information.
// Inputs preprocessed in complement coding are automatically normalized.
func (m *FuzzyART) complementCode(X []float64) []float64 {
	I := make([]float64, len(X)*2)
	// Create a new slice with double the length of the input slice
	for i, v := range X {
		// Copy the original value to the first half of the new slice
		I[i] = v
		// Calculate and store the complement (1 - X[i]) in the second half of the new slice
		I[i+len(X)] = 1 - v
	}

	return I
}

// min takes two slices of float64 values and returns a new slice
// containing the element-wise min of the two input slices.
// The function is used to calculate the fuzzy intersection between two vectors.
// The fuzzy intersection array is passed by reference
// to avoid unnecessary memory allocations.
func (m *FuzzyART) min(A, B, fuzzyIntersection []float64) {
	for i := range A {
		if A[i] < B[i] {
			fuzzyIntersection[i] = A[i]
		} else {
			fuzzyIntersection[i] = B[i]
		}
	}
}

// sum all the elements in a float64 slice.
// The function is used to calculate the L1 norm of a vector.
func (m *FuzzyART) sum(arr []float64) (norm float64) {
	for _, v := range arr {
		norm += v
	}

	return
}

// choice compute the choice function.
// Calculates the activation of a category based on the input vector.
// The fuzzyIntersection slice s passed by reference
// to avoid unnecessary memory allocations.
func (m *FuzzyART) choice(I, W, fuzzyIntersection []float64) (choice float64, fiNorm float64) {
	m.min(I, W, fuzzyIntersection)
	fiNorm = m.sum(fuzzyIntersection)
	choice = fiNorm / (m.alpha + m.sum(W))
	return
}

// categoryChoices implements the recognition field functionality
// by computing activation values for each category based on the input vector.
// The sorting process also implicitly handles lateral inhibition by prioritizing
// the category with the highest activation, thereby inhibiting others.
func (m *FuzzyART) categoryChoices(I []float64) (jList []int, fuzzyIntersectionList [][]float64, fuzzyIntersectionNormList []float64) {
	T := make([]float64, len(m.W))
	fuzzyIntersectionList = make([][]float64, len(m.W))
	fuzzyIntersectionNormList = make([]float64, len(m.W))

	for start := 0; start < len(m.W); start += m.batchSize {
		end := start + m.batchSize
		if end > len(m.W) {
			end = len(m.W)
		}

		m.wg.Add(1)
		m.workerPool <- struct{}{}

		go func(input []float64, categories [][]float64, startIndex int) {
			defer func() {
				<-m.workerPool
				m.wg.Done()
			}()

			for i, category := range categories {
				globalIndex := startIndex + i
				fuzzyIntersection := m.fuzzyIntersectionsPool.Get().([]float64)
				T[globalIndex], fuzzyIntersectionNormList[globalIndex] = m.choice(input, category, fuzzyIntersection)
				fuzzyIntersectionList[globalIndex] = fuzzyIntersection
			}
		}(I, m.W[start:end], start)
	}

	m.wg.Wait()

	// Create a list of category indices
	jList = make([]int, len(T))
	for i := range jList {
		jList[i] = i
	}

	// Sort category indices by activation values in descending order
	sort.SliceStable(jList, func(i, j int) bool {
		// In case of equal activation values, sort by category index
		// older categories have priority.
		if T[jList[i]] == T[jList[j]] {
			return jList[i] < jList[j]
		}
		return T[jList[i]] > T[jList[j]]
	})

	return
}

func (m *FuzzyART) match(fiNorm, iNorm float64) float64 {
	if fiNorm == 0 && iNorm == 0 {
		return 1
	} else {
		return fiNorm / iNorm
	}
}

// resonateOrReset implements the resonance or reset logic.
// If the best matching category passes the vigilance test (>= rho),
// its weights are updated to move closer to the input vector, facilitating learning.
// If it fails, the category is inhibited (temporarily ignored), and the next best category is tested,
// continuing until a suitable category is found or all are exhausted.
func (m *FuzzyART) resonateOrReset(I []float64, JList []int, fuzzyIntersectionList [][]float64, fiNormList []float64) (categoryWeights []float64, categoryIndex int) {
	iNorm := m.sum(I)

	for _, j := range JList {
		resonance := m.match(fiNormList[j], iNorm)
		if resonance >= m.rho {
			newW := make([]float64, len(m.W[j]))
			for k := range newW {
				newW[k] = m.beta*fuzzyIntersectionList[j][k] + (1-m.beta)*m.W[j][k]
			}

			m.W[j] = newW
			return newW, j
		}
	}

	// If no category meets the vigilance criterion, create a new category.
	// Fast commitment option, directly copy the input vector as the new category.
	m.W = append(m.W, I)

	return m.W[len(m.W)-1], len(m.W) - 1
}

func (m *FuzzyART) recover(fiList [][]float64) {
	for _, fi := range fiList {
		if fi == nil {
			break
		}
		m.fuzzyIntersectionsPool.Put(fi)
	}
}

//func (m *FuzzyART) WarmUp(inputLen int, size int) {
//	// Pre-warm the pool with enough arrays for all threads
//	for i := 0; i < size; i++ {
//		m.fuzzyIntersectionsPool.Put(make([]float64, inputLen*2))
//	}
//}

// Train implements the complete ART learning cycle
func (m *FuzzyART) Train(I []float64) ([]float64, int) {
	I = m.complementCode(I)
	jList, fiList, fiNormList := m.categoryChoices(I)
	categoryWeights, categoryIndex := m.resonateOrReset(I, jList, fiList, fiNormList)
	m.recover(fiList)
	return categoryWeights, categoryIndex
}

// Infer implements the recognition process with optional learning.
// It returns the weight vector of the best matching category and its index.
// If rtl (resonate-then-learn) is true, it updates the weights of the matching category.
func (m *FuzzyART) Infer(I []float64, rtl bool) ([]float64, int) {
	I = m.complementCode(I)
	jList, fiList, fiNormList := m.categoryChoices(I)
	if !rtl {
		m.recover(fiList)
		return m.W[jList[0]], jList[0]
	}
	categoryWeights, categoryIndex := m.resonateOrReset(I, jList, fiList, fiNormList)
	m.recover(fiList)
	return categoryWeights, categoryIndex
}

func (m *FuzzyART) Close() {
	close(m.workerPool)
}
