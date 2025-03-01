package art

import (
	"fmt"
	"runtime"
	"sort"
	"sync"
)

type Batch struct {
	A          []float64
	categories [][]float64
	startIndex int
}

type Worker struct {
	tPool *sync.Pool
	wg    *sync.WaitGroup
	alpha float64
	//resultsChan chan<- []*activation
	results []*activation
}

func NewWorker(features int, alpha float64, wg *sync.WaitGroup) *Worker { //, resultsChan chan<- []*activation) *Worker {
	return &Worker{
		tPool: &sync.Pool{
			New: func() interface{} {
				return &activation{
					fuzzyIntersection: make([]float64, features),
				}
			},
		},
		wg:      wg,
		alpha:   alpha,
		results: make([]*activation, 0),
	}
}

func (w *Worker) processBatch(
	batch Batch,
) {
	//results := make([]*activation, len(batch.categories))
	for j, W := range batch.categories {
		t := w.tPool.Get().(*activation)
		//t := &activation{
		//	fuzzyIntersection: make([]float64, len(W)),
		//}
		t.j = batch.startIndex + j
		w.choice(batch.A, W, t)
		w.results = append(w.results, t)
	}
	w.wg.Done()
	//w.results <- results
}

// choice compute the choice function.
// Calculates the activation of a category based on the input vector.
// The fuzzyIntersection slice s passed by reference to avoid unnecessary memory allocations.
func (w *Worker) choice(A, W []float64, activation *activation) {
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
		//activation.fuzzyIntersection[i] = math.Min(A[i], W[i])
		activation.fuzzyIntersectionNorm += activation.fuzzyIntersection[i]
	}

	activation.t = activation.fuzzyIntersectionNorm / (w.alpha + activation.wNorm)
}

func (w *Worker) recover() {
	for _, t := range w.results {
		w.tPool.Put(t)
	}
	w.results = make([]*activation, 0)
}

type FuzzyART struct {
	//workerPool chan Task
	workerPool []*Worker
	batchSize  int
	batchChan  chan Batch
	wg         *sync.WaitGroup
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

	fuzzyART := &FuzzyART{
		//workerPool: make(chan struct{}, runtime.NumCPU()),
		workerPool: make([]*Worker, runtime.NumCPU()),
		batchSize:  16,
		batchChan:  make(chan Batch, runtime.NumCPU()),
		wg:         &sync.WaitGroup{},
		tPool: &sync.Pool{
			New: func() interface{} {
				return &activation{
					fuzzyIntersection: make([]float64, inputLen*2),
				}
			},
		},
		rho:   rho,
		alpha: alpha,
		beta:  beta,
		W:     make([][]float64, 0),
	}

	// Create workers, one per CPU
	for i := 0; i < runtime.NumCPU(); i++ {
		fuzzyART.workerPool[i] = NewWorker(inputLen*2, alpha, fuzzyART.wg)
	}

	// Start the worker goroutines
	fuzzyART.startWorkers()

	return fuzzyART, nil
}

// startWorkers spawns worker goroutines, one per CPU
func (m *FuzzyART) startWorkers() {
	for i := range m.workerPool {
		//m.wg.Add(1)
		go func(workerIndex int) {
			//defer m.wg.Done()
			for batch := range m.batchChan {
				m.workerPool[workerIndex].processBatch(batch)
			}
		}(i)
	}
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

// choice compute the choice function.
// Calculates the activation of a category based on the input vector.
// The fuzzyIntersection slice s passed by reference to avoid unnecessary memory allocations.
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
		//activation.fuzzyIntersection[i] = math.Min(A[i], W[i])
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
// by computing activation values for each category based on the input vector.
// The sorting process also implicitly handles lateral inhibition by prioritizing
// the category with the highest activation, thereby inhibiting others.
func (m *FuzzyART) categoryChoices(A []float64) []*activation {
	T := make([]*activation, len(m.W))

	//mutex := &sync.Mutex{}

	for jStart := 0; jStart < len(m.W); jStart += m.batchSize {
		jEnd := jStart + m.batchSize
		if jEnd > len(m.W) {
			jEnd = len(m.W)
		}

		m.wg.Add(1)
		m.batchChan <- Batch{
			A:          A,
			categories: m.W[jStart:jEnd],
			startIndex: jStart,
		}

		//// acquire a worker
		//m.workerPool <- struct{}{}
		//
		//// spawn a goroutine to process a batch of categories
		//go func(A []float64, categories [][]float64, startIndex int) {
		//	defer func() {
		//		// release the worker
		//		<-m.workerPool
		//		m.wg.Done()
		//	}()
		//
		//	for j, W := range categories {
		//		u := m.tPool.Get().(*activation)
		//		u.j = startIndex + j
		//		m.choice(A, W, u)
		//		mutex.Lock()
		//		T[u.j] = u
		//		mutex.Unlock()
		//	}
		//}(A, m.W[jStart:jEnd], jStart)
	}

	m.wg.Wait()

	for _, worker := range m.workerPool {
		for _, t := range worker.results {
			T[t.j] = t
		}
	}

	// Sort category indices by activation values in descending order
	sort.SliceStable(T, func(a, b int) bool {
		// In case of equal activation values, sort by category index,
		// because older categories must have the priority.
		if T[a].t == T[b].t {
			return T[a].j < T[b].j
		}
		return T[a].t > T[b].t
	})

	return T
}

// match computes the match function.
// The match function calculates the resonance between the input vector and a category.
// The resonance is the ratio of the fuzzy intersection L1 norm to the input vector L1 norm.
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
// continuing until a suitable category is found or all are exhausted
// in which case a new category is created.
func (m *FuzzyART) resonateOrReset(
	A []float64,
	T []*activation,
) (categoryWeights []float64, categoryIndex int) {
	for _, t := range T {
		resonance := m.match(t.fuzzyIntersectionNorm, t.aNorm)
		if resonance >= m.rho {
			newW := make([]float64, len(A))
			for k := range newW {
				newW[k] = m.beta*t.fuzzyIntersection[k] + (1-m.beta)*m.W[t.j][k]
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
	for _, worker := range m.workerPool {
		worker.recover()
	}
	//for _, t := range T {
	//	m.tPool.Put(t)
	//}
}

// Train implements the complete ART learning cycle.
func (m *FuzzyART) Train(a []float64) ([]float64, int) {
	A := m.complementCode(a)
	T := m.categoryChoices(A)
	defer m.recover(T)
	return m.resonateOrReset(A, T)
}

// Infer implements the recognition process with optional learning.
// It returns the weight vector of the best matching category and its index.
// If learn is true, it updates the weights of the matching category.
func (m *FuzzyART) Infer(a []float64, learn bool) ([]float64, int) {
	A := m.complementCode(a)
	T := m.categoryChoices(A)
	defer m.recover(T)
	if !learn {
		return m.W[T[0].j], T[0].j
	}

	return m.resonateOrReset(A, T)
}

func (m *FuzzyART) Close() {
	close(m.batchChan)
	//close(m.workerPool)
}
