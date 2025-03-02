package art

import (
	"sort"

	"art/internal/simd"
)

type DefaultARTMAP2 struct {
	// Network parameters
	Rho          float64 // ρ Vigilance parameter
	RhoBar       float64 // ρ - baseline vigilance parameter
	Alpha        float64 // α - signal rule parameter, (0,1), α = 0 + maximizes code compression
	Beta         float64 // β - learning rate, [0,1], β = 1 implements fast learning
	Epsilon      float64 // ε - epsilon, match tracking, (-1,1), ε < 0 (MT–) codes inconsistent cases
	CAMRulePower float64 // Increased Gradient (IG) CAM rule converges to WTA as p→∞

	// Network structure
	M             int // M - dimension of input vectors
	NumCategories int // number of output classes
	NumOfClasses  int // number of output classes

	// Network weights
	w [][]float64 // w_j - weight vectors for coding nodes
	W []int       // W_j - map categories to output classes labels

	// Category mappings
	//MapField []int // maps coding nodes to output categories
}

// NewDefaultARTMAP2 creates a new Default ARTMAP 2 network with default parameters
func NewDefaultARTMAP2(inputDim, numClasses int) *DefaultARTMAP2 {
	// Initialize with default parameters as mentioned in the paper
	network := &DefaultARTMAP2{
		RhoBar:       0.9,  // Default value for maximal code compression
		Alpha:        0.01, // Default choice parameter
		Beta:         1.0,  // Fast learning
		Epsilon:      -0.001,
		CAMRulePower: 1.0,
		M:            inputDim,
		NumOfClasses: numClasses,
		w:            make([][]float64, 0),
		W:            make([]int, 0),
		//MapField:      []int{},
	}
	return network
}

// ComplementCode performs complement coding on an input vector
func (am *DefaultARTMAP2) ComplementCode(input []float64) []float64 {
	// Complement coding: concatenate input with its complement (1-input)
	coded := make([]float64, am.M*2)
	for i := 0; i < am.M; i++ {
		coded[i] = input[i]
		coded[i+am.M] = 1 - input[i]
	}
	return coded
}

func (am *DefaultARTMAP2) addNewCategory(newWeights []float64, k int) {
	am.w = append(am.w, newWeights)
	am.W = append(am.W, k)
}

type Activation struct {
	j                    int
	val                  float64
	distributedVal       float64
	fuzzyIntersection    []float64
	fuzzyIntersectionSum float64
}

// Activate categories
// B.7 and B.8:
// Calculate signals to committed coding nodes
// Sort the committed coding nodes with'0]
// T j > αM in descending order.
func (am *DefaultARTMAP2) choiceFunction(A []float64) []Activation {
	T := make([]Activation, 0)
	activationThreshold := am.Alpha * float64(am.M)

	fuzzyIntersection := make([]float64, len(A))

	for j := 0; j < len(am.w); j++ {
		// Use cross-platform optimized SIMD function
		fuzzyIntersectionSum := simd.FuzzyIntersectionSum(A, am.w[j], fuzzyIntersection)

		// Use cross-platform optimized SIMD function
		wSum := simd.SumFloat64(am.w[j])

		activationVal := fuzzyIntersectionSum + (1-am.Alpha)*(float64(am.M)-wSum)

		if activationVal < activationThreshold {
			// skip nodes with activation value less than threshold
			continue
		}

		T = append(T, Activation{
			j:                    j,
			val:                  activationVal,
			fuzzyIntersection:    fuzzyIntersection,
			fuzzyIntersectionSum: fuzzyIntersectionSum,
		})

		// new slice for the next iteration
		fuzzyIntersection = make([]float64, len(A))
	}

	// Sort category val values indices by val values in descending order
	sort.SliceStable(T, func(a, b int) bool {
		// In case of equal val values, sort by category index,
		// because older categories must have the priority.
		if T[a].val == T[b].val {
			return T[a].j < T[b].j
		}
		return T[a].val > T[b].val
	})

	return T
}

// B.11. Learning: Update coding weights
func (am *DefaultARTMAP2) updateWeights(A []float64, t Activation) {
	for i := 0; i < 2*am.M; i++ {
		//am.w[t.j][i] = am.Beta*math.Min(A[i], am.w[t.j][i]) + (1-am.Beta)*am.w[t.j][i]
		am.w[t.j][i] = am.Beta*t.fuzzyIntersection[i] + (1-am.Beta)*am.w[t.j][i]
	}
}

// Fit trains the network on a single input-output pair
func (am *DefaultARTMAP2) Fit(a []float64, k int) {
	A := am.ComplementCode(a)

	if len(am.w) == 0 {
		am.addNewCategory(A, k)
		return
	}

	// Winner-take-all coding during training
	// Find the best matching existing category or create a new one

	// coding field activation pattern (CAM): (yj)
	//y := make([]float64, len(am.w))

	T := am.choiceFunction(A)

	// B.5. Set vigilance ρ to its baseline value
	am.Rho = am.RhoBar

	matched := false

	// B.9. Search for a coding node J that meets the matching
	// criterion and predicts the correct output class K
	for _, t := range T {
		// B.9.a. Code: For the next sorted coding node that meets
		// the matching criterion
		resonance := t.fuzzyIntersectionSum / float64(am.M)
		if resonance >= am.Rho {
			// B.9.b. Output class prediction
			J := am.W[t.j]
			if J == k {
				// B.9.c. Correct prediction
				matched = true

				// B.11. Learning: Update coding weights
				am.updateWeights(A, t)

				// Distributed next-a test (specific to Default ARTMAP 2)
				// Test if the network would correctly classify the a with distributed val
				//prediction := am.Predict(A)
				//if prediction != k {
				//	am.Rho = resonance + am.Epsilon
				//	continue
				//}

				break
			} else {
				// B.9.d Match tracking: If the active code J fails to
				// predict the correct output class (σK = 0), raise vigilance:
				am.Rho = resonance + am.Epsilon
			}
		}
	}

	if !matched {
		// B.10. After unsuccessfully searching the sorted list,
		// add a committed node and return to Step B.4
		am.addNewCategory(A, k)
	}
}

// Predict predicts the class of an input using distributed val
func (am *DefaultARTMAP2) Predict(a []float64) int {
	if len(am.w) == 0 {
		return -1
	}

	A := am.ComplementCode(a)

	T := am.choiceFunction(A)

	return am.W[T[0].j]

	// Calculate distributed activations using increased-gradient CAM rule
	//am.calcDistributedActivation(A, T)

	//	// Count votes for each category
	//	categoryVotes := make([]float64, am.NumCategories)
	//	for _, t := range T {
	//		categoryVotes[am.W[t.j]] += t.distributedVal
	//	}
	//
	//	// Find the category with the highest vote
	//	maxVote := -1.0
	//	predictedClass := -1
	//	for k := 0; k < am.NumCategories; k++ {
	//		if categoryVotes[k] > maxVote {
	//			maxVote = categoryVotes[k]
	//			predictedClass = k
	//		}
	//	}
	//
	//	return predictedClass
}

//func (am *DefaultARTMAP2) calcDistributedActivation(A []float64, T []Activation) {
//	// increased-gradient CAM rule
//
//	// Normalize activations
//	totalActivation := 0.0
//	for _, t := range T {
//		totalActivation += t.val
//	}
//	for _, t := range T {
//		t.distributedVal = t.val / totalActivation
//	}
//}
