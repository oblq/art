package art

import (
	"math"
	"sort"
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

	// Network weights
	Weights          [][]float64 // w_j - weight vectors for coding nodes
	OutputCategories [][]float64

	// Category mappings
	MapField []int // maps coding nodes to output categories
}

// NewDefaultARTMAP2 creates a new Default ARTMAP 2 network with default parameters
func NewDefaultARTMAP2(inputDim, numCategories int) *DefaultARTMAP2 {
	// Initialize with default parameters as mentioned in the paper
	network := &DefaultARTMAP2{
		RhoBar:           0.0,  // Default value for maximal code compression
		Alpha:            0.01, // Default choice parameter
		Beta:             1.0,  // Fast learning
		Epsilon:          -0.001,
		CAMRulePower:     1.0,
		M:                inputDim,
		NumCategories:    numCategories,
		Weights:          make([][]float64, 0),
		OutputCategories: make([][]float64, 0),
		MapField:         []int{},
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

func (am *DefaultARTMAP2) addNewCategory(newWeights []float64, targetClass int) {
	am.Weights = append(am.Weights, newWeights)
	am.MapField = append(am.MapField, targetClass)
}

type Activation struct {
	j                    int
	val                  float64
	distributedVal       float64
	fuzzyIntersection    []float64
	fuzzyIntersectionSum float64
}

// B.7 and B.8:
// Calculate signals to committed coding nodes
// Sort the committed coding nodes with
// T j > αM in descending order.
func (am *DefaultARTMAP2) signalsToCommittedNodes(A []float64) []Activation {
	T := make([]Activation, 0)
	activationThreshold := am.Alpha * float64(am.M)

	for j := 0; j < len(am.Weights); j++ {
		fuzzyIntersection := make([]float64, len(A))
		fuzzyIntersectionSum := 0.0
		wSum := 0.0
		for i := 0; i < am.M; i++ {
			fuzzyIntersection[i] = math.Min(A[i], am.Weights[j][i])
			fuzzyIntersectionSum += fuzzyIntersection[i]
			wSum += am.Weights[j][i]
		}

		activationVal := fuzzyIntersectionSum + (1-am.Alpha)*(float64(am.M)-wSum)
		if activationVal < activationThreshold {
			continue
		}

		T = append(T, Activation{
			j:                    j,
			val:                  activationVal,
			fuzzyIntersection:    fuzzyIntersection,
			fuzzyIntersectionSum: fuzzyIntersectionSum,
		})
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
		//am.Weights[t.j][i] = am.Beta*math.Min(A[i], am.Weights[t.j][i]) + (1-am.Beta)*am.Weights[t.j][i]
		am.Weights[t.j][i] = am.Beta*t.fuzzyIntersection[i] + (1-am.Beta)*am.Weights[t.j][i]
	}
}

// Fit trains the network on a single input-output pair
func (am *DefaultARTMAP2) Fit(a []float64, k int) {
	A := am.ComplementCode(a)

	if len(am.Weights) == 0 {
		am.addNewCategory(A, k)
		return
	}

	// Winner-take-all coding during training
	// Find the best matching existing category or create a new one

	T := am.signalsToCommittedNodes(A)

	matched := false

	// B.9. Search for a coding node J that meets the matching
	// criterion and predicts the correct output class K
	for _, t := range T {
		// B.9.a. Code: For the next sorted coding node that meets
		// the matching criterion
		resonance := t.fuzzyIntersectionSum / float64(am.M)
		if resonance >= am.Rho {
			// B.9.b. Output class prediction
			J := am.MapField[t.j]
			if J == k {
				// B.9.c. Correct prediction

				// B.11. Learning: Update coding weights
				am.updateWeights(A, t)

				// Distributed next-a test (specific to Default ARTMAP 2)
				// Test if the network would correctly classify the a with distributed val
				prediction := am.Predict(A)
				if prediction != k {
					am.Rho = resonance + am.Epsilon
					continue
				}

				matched = true
				break
			} else {
				// B.9.d Match tracking: If the active code J fails to
				// predict the correct output class (σK = 0), raise vigilance:
				am.Rho = resonance + am.Epsilon
				continue
			}
		}
	}

	if !matched {
		// B.10. After unsuccessfully searching the sorted list,
		// add a committed node and return to Step B.4
		am.addNewCategory(A, k)
	}

	// B.5. Set vigilance ρ to its baseline value
	am.Rho = am.RhoBar
}

// Predict predicts the class of an input using distributed val
func (am *DefaultARTMAP2) Predict(a []float64) int {
	if len(am.Weights) == 0 {
		return -1
	}

	A := am.ComplementCode(a)

	T := am.signalsToCommittedNodes(A)

	// Calculate distributed activations using increased-gradient CAM rule
	am.calcDistributedActivation(A, T)

	// Count votes for each category
	categoryVotes := make([]float64, am.NumCategories)
	for _, t := range T {
		categoryVotes[am.MapField[t.j]] += t.distributedVal
	}

	// Find the category with the highest vote
	maxVote := -1.0
	predictedClass := -1
	for k := 0; k < am.NumCategories; k++ {
		if categoryVotes[k] > maxVote {
			maxVote = categoryVotes[k]
			predictedClass = k
		}
	}

	return predictedClass
}

func (am *DefaultARTMAP2) calcDistributedActivation(A []float64, T []Activation) {
	// increased-gradient CAM rule

	// Normalize activations
	totalActivation := 0.0
	for _, t := range T {
		totalActivation += t.val
	}
	for _, t := range T {
		t.distributedVal = t.val / totalActivation
	}
}
