package art

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

type FuzzyARTMAP struct {
	workerPool chan struct{}
	batchSize  int
	wg         sync.WaitGroup
	fiPool     *sync.Pool

	M int // Number of features
	//NumClasses      int         // Number of output classes
	W      [][]float64 // Weight matrix, F2 nodes
	Alpha  float64     // Choice parameter
	Beta   float64     // Learning rate
	RhoBar float64     // Baseline vigilance
	Rho    float64     // Vigilance parameter
	//Gamma           float64     // Match function parameter
	Epsilon float64 // Match tracking parameter
	//LambdaAttention float64     // Attention parameter
	//SearchCycles    int         // Number of search cycles
	mapField []string
}

func NewFuzzyARTMAP(m int, rho, alpha, beta float64) *FuzzyARTMAP {
	return &FuzzyARTMAP{
		workerPool: make(chan struct{}, runtime.NumCPU()),
		batchSize:  16,
		wg:         sync.WaitGroup{},
		fiPool: &sync.Pool{
			New: func() interface{} {
				return make([]float64, m*2)
			},
		},
		M: m,
		//NumClasses:      numClasses,
		Alpha:  alpha,
		Beta:   beta,
		RhoBar: rho,
		Rho:    rho,
		//Gamma:           1e-6,
		Epsilon: -1e-5,
		//LambdaAttention: 0.0,
		W:        make([][]float64, 0),
		mapField: make([]string, 0),
	}
}

func (f *FuzzyARTMAP) ComplementCode(data []float64) []float64 {
	complemented := make([]float64, 2*len(data))
	for i, val := range data {
		complemented[i] = val
		complemented[i+len(data)] = 1 - val
	}
	return complemented
}

func (f *FuzzyARTMAP) ChoiceFunction(a []float64) []float64 {
	t := make([]float64, len(f.W))
	//for j := 0; j < len(f.W); j++ {
	//	numerator := 0.0
	//	denominator := 0.0
	//	for i := 0; i < len(a); i++ {
	//		numerator += math.Min(a[i], f.W[j][i])
	//		denominator += f.W[j][i]
	//	}
	//	t[j] = numerator / (f.Alpha + denominator)
	//}
	//return t

	//type CategoryActivation struct {
	//	index int
	//	val   float64
	//}
	//tChan := make(chan CategoryActivation)
	//go func() {
	//	for categoryActivation := range tChan {
	//		t[categoryActivation.index] = categoryActivation.val
	//	}
	//}()

	for start := 0; start < len(f.W); start += f.batchSize {
		end := start + f.batchSize
		if end > len(f.W) {
			end = len(f.W)
		}

		f.wg.Add(1)
		// acquire a worker
		f.workerPool <- struct{}{}

		// spawn a goroutine to process a batch of categories
		go func(input []float64, categories [][]float64, startIndex int) {
			defer func() {
				// release the worker
				<-f.workerPool
				f.wg.Done()
			}()

			for relativeIndex, category := range categories {
				globalIndex := startIndex + relativeIndex

				numerator := 0.0
				denominator := 0.0
				for i := 0; i < len(input); i++ {
					numerator += math.Min(input[i], category[i])
					//numerator += math.Max(0, input[i]-category[i])
					denominator += category[i]
				}
				//t[globalIndex] = numerator + (1-f.Alpha)*(float64(f.M)-denominator)
				t[globalIndex] = numerator / (f.Alpha + denominator)
				//tChan <- CategoryActivation{
				//	index: globalIndex,
				//	//val:   numerator / (f.Alpha + denominator),
				//	val: numerator / ((1 - f.Alpha) * (float64(len(input)) - denominator)),
				//}
			}
		}(a, f.W[start:end], start)
	}

	f.wg.Wait()

	return t
}

func (f *FuzzyARTMAP) UpdateWeights(j int, a []float64) {
	for i := 0; i < 2*f.M; i++ {
		f.W[j][i] = f.Beta*math.Min(a[i], f.W[j][i]) + (1-f.Beta)*f.W[j][i]
	}
}

func (f *FuzzyARTMAP) Fit(a []float64, class string) {
	k := class
	a = f.ComplementCode(a)

	if len(f.W) == 0 {
		f.addNewCategory(a, k)
		return
	}

	T := f.ChoiceFunction(a)

	// Create a list of category activation values indices
	jList := make([]int, len(T))
	for i := range jList {
		jList[i] = i
	}

	// Sort category activation values indices by activation values in descending order
	sort.SliceStable(jList, func(i, j int) bool {
		// In case of equal activation values, sort by category index,
		// because older categories must have the priority.
		if T[jList[i]] == T[jList[j]] {
			return jList[i] < jList[j]
		}
		return T[jList[i]] > T[jList[j]]
	})

	match := false

	for _, j := range jList {
		if T[j] < f.Alpha*float64(f.M) {
			break // Stop if T is too small
		}

		resonance := f.MatchFunction(a, j)
		if resonance >= f.Rho {
			if f.checkCategoryMatch(j, k) {
				f.UpdateWeights(j, a)
				match = true
				break
			} else {
				//f.Rho = resonance + f.Epsilon // Match tracking
			}
		}
	}

	if !match {
		f.addNewCategory(a, k)
	}

	//for _, j := range jList {
	//	tj := T[j]
	//	//j := f.findIndex(t, tj)
	//	if tj <= f.Alpha*float64(f.M)*3.5 {
	//		continue
	//	}
	//
	//	xSum := f.calculateXSum(a, j)
	//
	//	if (xSum+f.Gamma)/(sum(a)+f.Gamma) >= f.Rho {
	//		f.SearchCycles++
	//		if f.checkCategoryMatch(j, k) {
	//			match = true
	//			f.UpdateWeights(j, a)
	//			f.NodeList = append(f.NodeList, j)
	//			break
	//		} else {
	//			f.Rho = (xSum+f.Gamma)/(sum(a)+f.Gamma) + f.Epsilon
	//			// Simplified ART_depleter_new logic
	//			f.updateAttentionParameters(a, j)
	//		}
	//	}
	//}
	//
	//if !match {
	//	f.C++
	//	f.W = append(f.W, a)
	//	f.expandMapField(k)
	//	f.NodeList = append(f.NodeList, f.C-1)
	//}

}

func (f *FuzzyARTMAP) Predict(a []float64) ([]float64, string) {
	if len(f.W) == 0 {
		return nil, "" // No categories learned yet
	}

	a = f.ComplementCode(a)

	T := f.ChoiceFunction(a)

	// Find the index of the maximum T value
	maxIndex := 0
	maxValue := T[0]
	for i, activation := range T {
		if activation > maxValue {
			maxValue = activation
			maxIndex = i
		}
	}

	return f.W[maxIndex], f.mapField[maxIndex]

	// Check if the winning category passes vigilance
	//if f.MatchFunction(a, maxIndex) >= f.Rho {
	// Find the predicted class (the one with the highest activation in the map field)
	//predictedClass := 0
	//maxActivation := f.mapField[maxIndex][0]
	//for class := 1; class < f.NumClasses; class++ {
	//	if f.mapField[maxIndex][class] > maxActivation {
	//		maxActivation = f.mapField[maxIndex][class]
	//		predictedClass = class
	//	}
	//}
	//return predictedClass
	//}

	// If no category passes vigilance, return -1 (or you could define some default behavior)
	//return nil, ""

	//// Create a list of category activation values indices
	//jList := make([]int, len(T))
	//for i := range jList {
	//	jList[i] = i
	//}
	//
	//// Sort category activation values indices by activation values in descending order
	//sort.SliceStable(jList, func(i, j int) bool {
	//	// In case of equal activation values, sort by category index,
	//	// because older categories must have the priority.
	//	if T[jList[i]] == T[jList[j]] {
	//		return jList[i] < jList[j]
	//	}
	//	return T[jList[i]] > T[jList[j]]
	//})
	//
	//return f.W[jList[0]], f.mapField[jList[0]]
}

func (f *FuzzyARTMAP) MatchFunction(a []float64, j int) float64 {
	numerator := 0.0
	denominator := 0.0
	for i := 0; i < len(a); i++ {
		numerator += math.Min(a[i], f.W[j][i])
		//numerator += math.Max(0, a[i]-f.W[j][i])
		denominator += a[i]
	}

	return numerator / denominator
	//return numerator / float64(f.M)
	//return numerator / (float64(len(a)) - denominator)
	//return numerator / denominator
}

func (f *FuzzyARTMAP) addNewCategory(a []float64, k string) {
	f.W = append(f.W, a)
	f.mapField = append(f.mapField, k)
}

func (f *FuzzyARTMAP) checkCategoryMatch(j int, k string) bool {
	return f.mapField[j] == k
}

//func (f *FuzzyARTMAP) Classify(inputVector []float64) int {
//	if f.C == 0 {
//		return -1 // No categories learned yet
//	}
//
//	t := f.ChoiceFunction(inputVector)
//	maxIndex := 0
//	maxValue := t[0]
//	for i, v := range t {
//		if v > maxValue {
//			maxValue = v
//			maxIndex = i
//		}
//	}
//
//	// Find the class with the highest activation in the map field
//	classActivations := make([]float64, f.NumClasses)
//	for i := 0; i < f.NumClasses; i++ {
//		classActivations[i] = f.mapField[maxIndex][i]
//	}
//
//	maxClass := 0
//	maxClassValue := classActivations[0]
//	for i, v := range classActivations {
//		if v > maxClassValue {
//			maxClassValue = v
//			maxClass = i
//		}
//	}
//
//	return maxClass
//}

//func (f *FuzzyARTMAP) calculateXSum(a []float64, j int) float64 {
//	xSum := 0.0
//	for i := 0; i < 2*f.M; i++ {
//		xSum += math.Min(math.Max(f.W[j][i]-f.LambdaAttention*0, 0),
//			math.Max(a[i]-f.LambdaAttention*0, 0))
//	}
//	return xSum
//}

//func (f *FuzzyARTMAP) updateAttentionParameters(a []float64, j int) {
//	// Simplified attention update logic
//	for i := 0; i < 2*f.M; i++ {
//		f.W[j][i] = math.Max(f.W[j][i]-f.LambdaAttention, 0)
//	}
//}

//func (f *FuzzyARTMAP) findIndex(slice []float64, value float64) int {
//	for i, v := range slice {
//		if v == value {
//			return i
//		}
//	}
//	return -1
//}

func sum(slice []float64) float64 {
	sum := 0.0
	for _, v := range slice {
		sum += v
	}
	return sum
}

//
//type FuzzyARTMAP struct {
//	workerPool chan struct{}
//	batchSize  int
//	wg         sync.WaitGroup
//	fiPool     *sync.Pool
//
//	// Vigilance parameter - controls category granularity
//	// Recommended value: 0.8
//	// Range: 0.0 to 1.0
//	// Typical Values: 0.5 to 0.9
//	// Purpose: Determines the strictness of category matching. Higher values mean stricter matching criteria.
//	// Adjustment:
//	// Increase rho to make the model more selective, creating more categories.
//	// Decrease rho to allow more generalization, creating fewer categories.
//	rho float64
//
//	// Choice parameter - influences category competition
//	// Recommended value: 0.01
//	// Range: > 0.0
//	// Typical Values: 0.0001 to 0.1
//	// Purpose: Influences the activation function, affecting the competition among categories.
//	// Adjustment:
//	// Higher alpha values generates lower categories competition (creates fewer).
//	// Lower alpha values (closer to 0) higher categories competition (creates more).
//	alpha float64
//
//	// Learning rate - controls weight update speed
//	// When beta == 1 is called fast-learning, it becomes sensitive to noise.
//	// Lower β values provide more gradual, stable learning but require more training iterations.
//	// Recommended value: 1.0
//	// Range: 0.0 to 1.0
//	// Purpose: Controls the rate at which the weights are updated during training.
//	// Adjustment:
//	// Increase beta for faster learning, which can be useful for rapidly changing environments.
//	// Decrease beta for more stable learning, which can be beneficial for more stable environments.
//	beta float64
//
//	// Weight matrix - stores category prototypes
//	W [][]float64
//}
//
//func NewFuzzyARTMAP(inputLen int, rho float64, alpha float64, beta float64) (*FuzzyARTMAP, error) {
//	if rho < 0 || rho > 1 {
//		return nil, fmt.Errorf("vigilance parameter (rho) must be between 0 and 1, got %f", rho)
//	}
//	if alpha <= 0 {
//		return nil, fmt.Errorf("choice parameter (alpha) must be positive, got %f", alpha)
//	}
//	if beta <= 0 || beta > 1 {
//		return nil, fmt.Errorf("learning rate (beta) must be between 0 and 1, got %f", beta)
//	}
//
//	return &FuzzyARTMAP{
//		workerPool: make(chan struct{}, runtime.NumCPU()),
//		batchSize:  16,
//		wg:         sync.WaitGroup{},
//		fiPool: &sync.Pool{
//			New: func() interface{} {
//				return make([]float64, inputLen*2)
//			},
//		},
//		rho:   rho,
//		alpha: alpha,
//		beta:  beta,
//		W:     make([][]float64, 0),
//	}, nil
//}
//
//// complementCode creates complement-coded representation of input vector.
//// Complement coding is a common preprocessing step in ART models
//// to prevent the "category proliferation problem."
//// Complement coding achieve normalization while preserving amplitude information.
//// Inputs preprocessed in complement coding are automatically normalized.
//func (m *FuzzyARTMAP) complementCode(X []float64) []float64 {
//	// Create a new slice with double the length of the input slice
//	I := make([]float64, len(X)*2)
//	for i, v := range X {
//		// Copy the original value to the first half of the new slice
//		I[i] = v
//		// Calculate and store the complement (1 - X[i]) in the second half of the new slice
//		I[i+len(X)] = 1 - v
//	}
//
//	return I
//}
//
//// min takes two slices of float64 values and returns a new slice
//// containing the element-wise min of the two input slices.
//// The function is used to calculate the fuzzy intersection between two vectors.
//// The fuzzy intersection slice is passed by reference to avoid unnecessary memory allocations.
//func (m *FuzzyARTMAP) min(A, B, fuzzyIntersection []float64) {
//	for i := range A {
//		if A[i] < B[i] {
//			fuzzyIntersection[i] = A[i]
//		} else {
//			fuzzyIntersection[i] = B[i]
//		}
//	}
//}
//
//// sum all the elements in a float64 slice.
//// The function is used to calculate the L1 norm of a vector.
//func (m *FuzzyARTMAP) sum(arr []float64) (norm float64) {
//	for _, v := range arr {
//		norm += v
//	}
//
//	return
//}
//
//// choice compute the choice function.
//// Calculates the activation of a category based on the input vector.
//// The fuzzyIntersection slice s passed by reference to avoid unnecessary memory allocations.
//func (m *FuzzyARTMAP) choice(I, W, fuzzyIntersection []float64) (choice float64, fiNorm float64) {
//	m.min(I, W, fuzzyIntersection)
//	fiNorm = m.sum(fuzzyIntersection)
//	choice = fiNorm / (m.alpha + m.sum(W))
//	return
//}
//
//// categoryChoices implements the recognition field functionality
//// by computing activation values for each category based on the input vector.
//// The sorting process also implicitly handles lateral inhibition by prioritizing
//// the category with the highest activation, thereby inhibiting others.
//func (m *FuzzyARTMAP) categoryChoices(I []float64) (jList []int, fiList [][]float64, fiNormList []float64) {
//	// Categories activations
//	T := make([]float64, len(m.W))
//	// Fuzzy intersections
//	fiList = make([][]float64, len(m.W))
//	// Fuzzy intersection norms
//	fiNormList = make([]float64, len(m.W))
//
//	for start := 0; start < len(m.W); start += m.batchSize {
//		end := start + m.batchSize
//		if end > len(m.W) {
//			end = len(m.W)
//		}
//
//		m.wg.Add(1)
//		// acquire a worker
//		m.workerPool <- struct{}{}
//
//		// spawn a goroutine to process a batch of categories
//		go func(input []float64, categories [][]float64, startIndex int) {
//			defer func() {
//				// release the worker
//				<-m.workerPool
//				m.wg.Done()
//			}()
//
//			for i, category := range categories {
//				globalIndex := startIndex + i
//				// Get a slice from the pool, rotating already allocated slices for efficiency
//				fuzzyIntersection := m.fiPool.Get().([]float64)
//				T[globalIndex], fiNormList[globalIndex] = m.choice(input, category, fuzzyIntersection)
//				fiList[globalIndex] = fuzzyIntersection
//			}
//		}(I, m.W[start:end], start)
//	}
//
//	m.wg.Wait()
//
//	// Create a list of category indices
//	jList = make([]int, len(T))
//	for i := range jList {
//		jList[i] = i
//	}
//
//	// Sort category indices by activation values in descending order
//	sort.SliceStable(jList, func(i, j int) bool {
//		// In case of equal activation values, sort by category index,
//		// because older categories must have the priority.
//		if T[jList[i]] == T[jList[j]] {
//			return jList[i] < jList[j]
//		}
//		return T[jList[i]] > T[jList[j]]
//	})
//
//	return
//}
//
//// match computes the match function.
//// The match function calculates the resonance between the input vector and a category.
//// The resonance is the ratio of the fuzzy intersection L1 norm to the input vector L1 norm.
//func (m *FuzzyARTMAP) match(fiNorm, iNorm float64) float64 {
//	if fiNorm == 0 && iNorm == 0 {
//		return 1
//	} else {
//		return fiNorm / iNorm
//	}
//}
//
//// resonateOrReset implements the resonance or reset logic.
//// If the best matching category passes the vigilance test (>= rho),
//// its weights are updated to move closer to the input vector, facilitating learning.
//// If it fails, the category is inhibited (temporarily ignored), and the next best category is tested,
//// continuing until a suitable category is found or all are exhausted
//// in which case a new category is created.
//func (m *FuzzyARTMAP) resonateOrReset(
//	I []float64,
//	jList []int,
//	fiList [][]float64,
//	fiNormList []float64,
//) (categoryWeights []float64, categoryIndex int) {
//	iNorm := m.sum(I)
//
//	for _, j := range jList {
//		resonance := m.match(fiNormList[j], iNorm)
//		if resonance >= m.rho {
//			newW := make([]float64, len(m.W[j]))
//			for k := range newW {
//				newW[k] = m.beta*fiList[j][k] + (1-m.beta)*m.W[j][k]
//			}
//
//			m.W[j] = newW
//			return newW, j
//		}
//	}
//
//	// If no category meets the vigilance criterion, create a new category.
//	// Fast commitment option, directly copy the input vector as the new category.
//	m.W = append(m.W, I)
//
//	return m.W[len(m.W)-1], len(m.W) - 1
//}
//
//// recover returns the fuzzy intersection slices to the pool for reuse.
//func (m *FuzzyARTMAP) recover(fiList [][]float64) {
//	for _, fi := range fiList {
//		m.fiPool.Put(fi)
//	}
//}
//
//// Train implements the complete ART learning cycle.
//func (m *FuzzyARTMAP) Train(I []float64) ([]float64, int) {
//	I = m.complementCode(I)
//	jList, fiList, fiNormList := m.categoryChoices(I)
//	defer m.recover(fiList)
//	return m.resonateOrReset(I, jList, fiList, fiNormList)
//}
//
//// Infer implements the recognition process with optional learning.
//// It returns the weight vector of the best matching category and its index.
//// If learn is true, it updates the weights of the matching category.
//func (m *FuzzyARTMAP) Infer(I []float64, learn bool) ([]float64, int) {
//	I = m.complementCode(I)
//	jList, fiList, fiNormList := m.categoryChoices(I)
//	defer m.recover(fiList)
//	if !learn {
//		return m.W[jList[0]], jList[0]
//	}
//
//	return m.resonateOrReset(I, jList, fiList, fiNormList)
//}
//
//func (m *FuzzyARTMAP) Close() {
//	close(m.workerPool)
//}
